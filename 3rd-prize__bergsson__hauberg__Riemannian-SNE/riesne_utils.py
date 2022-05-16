import sys
import math
import copy
from time import time
from tqdm import trange
from rich.progress import track

import numpy as np
import torch
import torchplot as plt
import jax
import jax.numpy as jnp

from scipy import linalg
from scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


MACHINE_EPSILON = np.finfo(np.double).eps
EPSILON_DBL = 1e-8
PERPLEXITY_TOLERANCE = 1e-5


def _riesne_binary_search_perplexity_log_scale(sqdistances, logH0, no_dims, desired_perplexity):
    # Maximum number of binary search steps
    n_steps = 100

    n_samples = sqdistances.shape[0]
    n_neighbors = sqdistances.shape[1]

    t_sum = 0.0

    desired_entropy = math.log2(desired_perplexity)

    logP = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    for i in track(range(n_samples), description=f'[Rie-SNE] Binary search for perplexity ...'):
        t_min = -np.inf
        t_max = np.inf
        t = 1.0

        # Binary search for variance t
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities over all data
            for j in range(n_neighbors):
                if j != i:
                    logP[i, j] = (-no_dims/2) * np.log(2*np.pi*t) + logH0[i,j] - (sqdistances[i, j] / (2*t))

            logP[i] -= logsumexp(np.delete(logP[i], i))
            logP[i, i] = 0.0

            log_sum_Pi_entropy = logsumexp(logP[i], b=(-(logP[i]/np.log(2))))

            entropy = np.exp(log_sum_Pi_entropy)
            entropy_diff = entropy - desired_entropy

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff < 0.0:
                t_min = t
                if t_max == np.inf:
                    t *= 2.0
                else:
                    t = (t + t_max) / 2.0
            else:
                t_max = t
                if t_min == -np.inf:
                    t /= 2.0
                else:
                    t = (t + t_min) / 2.0

        t_sum += t

    print(f"[Rie-SNE] Mean variance:{t_sum / n_samples}")

    P = np.exp(logP)
    P[range(n_samples), range(n_samples)] = 0.0
    return P


def _riesne_binary_search_perplexity(sqdistances, H0, no_dims, desired_perplexity):
    # Maximum number of binary search steps
    n_steps = 100

    n_samples = sqdistances.shape[0]
    n_neighbors = sqdistances.shape[1]

    t_sum = 0.0

    desired_entropy = math.log2(desired_perplexity)

    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    for i in (tr := trange(n_samples)):
        tr.set_description(f'[Rie-SNE] Binary search for perplexity ...')
        t_min = -np.inf
        t_max = np.inf
        t = 1.0

        ts = []
        es = []
        # Binary search for variance t
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities over all data
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i:
                    P[i, j] = ((2*np.pi*t)**(-(no_dims/2))) * H0[i,j] * math.exp(-sqdistances[i, j] / (2*t))
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL

            sum_Pi_entropy = 0.0
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                if P[i,j] != 0.0:
                    sum_Pi_entropy += P[i,j] * math.log2(P[i, j])

            entropy = - sum_Pi_entropy
            entropy_diff = entropy - desired_entropy

            ts.append(t)
            es.append(entropy)

            if math.fabs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff < 0.0:
                t_min = t
                if t_max == np.inf:
                    t *= 2.0
                else:
                    t = (t + t_max) / 2.0
            else:
                t_max = t
                if t_min == -np.inf:
                    t /= 2.0
                else:
                    t = (t + t_min) / 2.0

        t_sum += t

    print(f"[Rie-SNE] Mean variance:{t_sum / n_samples}")

    return P


def _compute_p_values(model, X, perplexity):
    X = torch.from_numpy(X).float()
    if X.dim() == 2:
        _, m = X.shape
    elif X.dim() == 3:
        _, m, _= X.shape

    dissimilarity_matrix = model.compute_pairwise_geodesic_distances(X)
    logH0 = model.compute_log_riemannian_volume_measure_ratios(X)

    """
    ====================
    Compute conditional probabilities using binary search for variance values 
   ====================
    """
    P = _riesne_binary_search_perplexity_log_scale(np.square(dissimilarity_matrix.cpu().numpy()), logH0.cpu().numpy(), m, perplexity)

    P = P + P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P /= sum_P
    P = np.maximum(squareform(P), MACHINE_EPSILON)

    return P


def _kl_divergence_func(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(0, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_func_spherical(params, P, degrees_of_freedom, n_samples, n_components, kappa):
    X_embedded = params.reshape(n_samples, n_components)
    X_embedded_jax = jnp.asarray(X_embedded)

    def kl_divergence_compute(P, X_embedded_jax, kappa):
        # Maximum likelihood estimate for kappa
        #R = jnp.linalg.norm(jnp.sum(X_embedded_jax, axis=1)).val.item() / n_samples
        #kappa_hat = (R * (n_components - R**2)) / (1 - R**2)
        #print(f'kappa hat is: {kappa_hat}')


        dist = jnp.sum((X_embedded_jax[None,:] - X_embedded_jax[:, None])**2, -1)
        pdf = jnp.exp(-kappa * dist)

        # Q is a a spherical distribtuion: the von Mises–Fisher distribution
        Q = jnp.maximum(pdf / jnp.sum(pdf), MACHINE_EPSILON)
        P = jnp.asarray(squareform(P))

        # Objective: C (Kullback-Leibler divergence of P and Q)
        return jnp.sum(P * jnp.log(jnp.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    derivative_fn = jax.value_and_grad(kl_divergence_compute, argnums=1)
    derivative_fn_kappa = jax.value_and_grad(kl_divergence_compute, argnums=2)

    kl_divergence, grad_immuteable = derivative_fn(P, X_embedded_jax, kappa)
    _, kappa_grad = derivative_fn_kappa(P, X_embedded_jax, kappa)

    grad = np.asarray(grad_immuteable).copy()

    return kl_divergence.item(), grad.ravel(), kappa_grad.item()

def _gradient_descent(objective, p0, it, n_iter, model,
                      n_iter_check=1, n_iter_without_progress=100,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-4, verbose=0, args=None, kwargs=None, init='random', kappa_init=1.0):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    if init == 'sphere':
        kwargs['kappa'] = kappa_init
    if init == 'rbm':
        kwargs['model'] = model

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    n_samples, n_components = args[2], args[3]

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0

        if init == 'sphere':
            error, grad, kappa_grad = objective(p, *args, **kwargs)
        else:
            error, grad = objective(p, *args, **kwargs)

        average_grad_norm = linalg.norm(grad) / n_samples
        max_grad_norm = np.max(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if init == 'sphere':
            p = np.reshape(p, (n_samples, n_components))
            p /= np.expand_dims(np.linalg.norm(p, axis=1), 1)
            p = p.ravel()

            kwargs['kappa'] -= 0.00001*kappa_grad

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                print(f'Reason 1: Break out of grad descent in iteration: {i}')
                break
            if max_grad_norm <= min_grad_norm:
                print(f'Reason 2: Break out of grad descent in iteration: {i}')
                break

    if 'kappa' in kwargs:
        kappa = kwargs['kappa']
    else:
        kappa = 1.0

    return p, error, i, kappa
