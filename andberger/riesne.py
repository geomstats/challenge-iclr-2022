import math
import sys

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

    """
    Riemannian Stochastic Neighbor Embedding.

    Parameters
    ----------
    model: RiesneModel
        The high-dimensional Riemannian manifold that the data to be visualized is lying on. 
        The model must derive from the abstract base model `RiesneModel` and implement the 
        two following methods:
        - compute_pairwise_geodesic_distances:
            Compute a matrix of pairwise distances by finding geodesics between all 
            pairwise points on the manifold and computing their lengths.
        - compute_log_riemannian_volume_measure_ratios
            Compute a matrix of pairwise Riemannian volume measure ratios, i.e.:
                H0(x,l) = (det(G(x)) / det(G(l)))^(1/2)
                Where G(p) for point p on the manifold is the metric matrix evaluated at point p.

    n_components : int, default=2
        Dimension of the embedded space. Use n_components=3 for spherical embeddings.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float or 'auto', default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.

    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    """
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
            init="random", 
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
        self.init = init

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
