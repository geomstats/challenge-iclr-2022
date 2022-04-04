import math
import sys
import copy
from time import time
from tqdm import trange

import numpy as np
import torch
import jax
import jax.numpy as jnp

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from scipy import linalg
from scipy.spatial.distance import pdist, squareform
import riesne_utils as ru


class Riesne(BaseEstimator):
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
            self, 
            model=None, 
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0, 
            learning_rate=200.0, 
            n_iter=1000,
            n_iter_without_progress=100, 
            min_grad_norm=1e-7,
            metric="euclidean", 
            init="random", 
            random_state=None, 
            method='kmeans', 
            angle=0.5,
            n_jobs=None, 
            n_clusters=75,
            dissimilarity_matrix=None, 
            logH0=None
        ):
        self.model = model
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

    def fit(self, X):
        """Fit"""
        n_samples = X.shape[0]

        P = ru._compute_p_values(
            self.model, 
            X, 
            self.perplexity, 
        )

        if self.init == "sphere":
            random_state = np.random.RandomState(seed=1234)
            X_embedded = 1e-4 * random_state.randn(len(X), 3).astype(np.float32)
            # project data onto sphere
            X_embedded /= np.expand_dims(np.linalg.norm(X_embedded, axis=1), 1)
        else:
            random_state = np.random.RandomState(seed=1234)
            X_embedded = 1e-4 * random_state.randn(len(X), 2).astype(np.float32)

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self.riesne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

    def riesne(self, P, degrees_of_freedom, n_samples, X_embedded):
        """Runs Rie-SNE."""
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
            "init": self.init,
            "kappa_init": 1.0,
            "model": self.model,
        }

        if self.init == 'sphere':
            obj_func = ru._kl_divergence_func_spherical
        else:
            obj_func = ru._kl_divergence_func

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration

        params, kl_divergence, it, kappa = ru._gradient_descent(obj_func, params, **opt_args)
        print(f'[Rie-SNE] KL divergence after {it+1} iterations with early exaggeration: {kl_divergence}')

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        opt_args['n_iter'] = self.n_iter
        opt_args['it'] = it + 1
        opt_args['momentum'] = 0.8
        opt_args['n_iter_without_progress'] = self.n_iter_without_progress
        opt_args['kappa_init'] = kappa

        params, kl_divergence, it, _ = ru._gradient_descent(obj_func, params, **opt_args)
        print(f'[Rie-SNE] KL divergence after {it + 1} iterations: {kl_divergence}')

        X_embedded = params.reshape(n_samples, self.n_components)

        return X_embedded

    def fit_transform(self, X):
        return self.fit(X)
