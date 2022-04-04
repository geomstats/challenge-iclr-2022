from abc import ABC, abstractmethod

class RiesneModel(ABC):
    @abstractmethod
    def compute_pairwise_geodesic_distances(self, points):
        pass

    @abstractmethod
    def compute_log_riemannian_volume_measure_ratios(self, points):
        pass
