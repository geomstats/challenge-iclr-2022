import torch
import torchplot as plt
import numpy as np

from models.riesne_model import RiesneModel
import geomstats.geometry.euclidean as geomstatsEuclidean

from sklearn.metrics import pairwise_distances


class Euclidean(RiesneModel):
    def __init__(self, dim):
        super(Euclidean, self).__init__()
        self.dim = dim
        self.manifold = geomstatsEuclidean.Euclidean(dim=self.dim)
        self.metric = geomstatsEuclidean.EuclideanMetric(self.dim)

    def compute_pairwise_geodesic_distances(self, X):
        return torch.from_numpy(pairwise_distances(X, metric=self.metric.dist)).float()

    def compute_log_riemannian_volume_measure_ratios(self, X):
        """
        For euclidean space, we can approximate the Riemannian volume measure ratios as the identity matrix, 
        but to compute the ratios we could do something like:
            ratios_function = lambda x, l : 
                np.sqrt(
                    np.linalg.det(self.metric.metric_matrix(base_point=x)) / 
                    np.linalg.det(self.metric.metric_matrix(base_point=l))
                )
            return torch.from_numpy(np.log(pairwise_distances(X, metric=ratios_function))).float()
        """
        n = X.shape[0]
        H0 = torch.ones((n, n)).float()
        H0[range(n), range(n)] = 0.0
        return H0
