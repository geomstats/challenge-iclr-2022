import geomstats.backend as gs


def dist_broadcast_l2(self, point_a, point_b):
    """Compute the geodesic distance between points on NFoldManifold.
    If n_samples_a == n_samples_b then dist is the element-wise
    distance result of a point in points_a with the point from
    points_b of the same index. If n_samples_a not equal to
    n_samples_b then dist is the result of applying geodesic
    distance for each point from points_a to all points from
    points_b. This simply computes the square-root of the sum
    of the squared distances of each component manifold.
    Parameters
    ----------
    point_a : array-like, shape=[n_samples_a, dim]
        Set of points in the Poincare ball.
    point_b : array-like, shape=[n_samples_b, dim]
        Second set of points in the Poincare ball.
    Returns
    -------
    dist : array-like,
        shape=[n_samples_a, dim] or [n_samples_a, n_samples_b, dim]
        Geodesic distance between the two points.
    """
    distances = []
    for i in range(self.n_copies):
      distances.append(
          self.base_manifold.metric.dist_broadcast(
              point_a[:, i], point_b[:, i])
      )
    distances = gs.stack(distances, axis=0)
    distance = gs.sqrt(gs.sum(distances**2, axis=0))

    return distance

