import torch
import torchplot as plt
import numpy as np

from models.riesne_model import RiesneModel
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric

from sklearn.metrics import pairwise_distances
import seaborn as sns
sns.set()


class Sphere(RiesneModel):
    def __init__(self, dim):
        super(Sphere, self).__init__()
        self.dim = dim
        self.manifold = Hypersphere(dim=self.dim)
        self.metric = HypersphereMetric(self.dim)

    def compute_pairwise_geodesic_distances(self, X):
        return torch.from_numpy(pairwise_distances(X, metric=self.metric.dist)).float()

    def compute_log_riemannian_volume_measure_ratios(self, X):
        """
        For the sphere, we can approximate the Riemannian volume measure ratios as the identity matrix, 
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

    def plot_points(self, transformed, y_data, title=''):
        from mpl_toolkits.mplot3d import Axes3D
        sns.set_style("white")
        sns.set_palette("rocket")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        labels = y_data
        for label in labels.unique():
            idx = labels == label
            points = transformed[idx]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        ax.set_axis_off()

        N=200
        stride=2
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, alpha=0.15)

        fig.suptitle(title, fontsize=40)
        plt.show()
