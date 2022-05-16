import numpy as np
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import NFoldManifold


def generate_data(manifold, num_neurons=100, num_data=300, sigma=2.0,
                  scale=10.0, seed=232987):
    """Generate responses of neurons with (Gaussian bump like) RFs on a manifold.
    Parameters
    ----------
    num_neurons : integer
        Number of neurons in simulation.
    num_data : integer
        Number of data points in simulation.
    sigma : float
        Width of the neuron's RF on the manifold.
    scale : float
        Mean firing rate of the neurons (proportional to SNR).
    seed : integer
        Random seed used in simulations.
    Returns
    -------
    centers : array-like, shape=[num_neurons, ...]
        Neurons' RF centers on the manifold.
    latents : array-like, shape=[num_neurons, ...]
        Random points on the manifold.
    responses : array-like, shape=[num_neurons, num_data]
        Neurons' responses to latents.
    responses_clean : array-like, shape=[num_neurons, num_data]
        Neurons' responses to latents without poisson noise.
    """
    # Generate random latents (points on manifold) and neurons (RF centers on manifold).
    gs.random.seed(seed)
    centers = manifold.random_point(n_samples=num_neurons)
    latents = manifold.random_point(n_samples=num_data)

    # Compute responses
    distances = manifold.metric.dist_broadcast(centers, latents)
    responses_clean = gs.exp(- distances**2 / sigma) * scale

    # Todo:(add poisson to gs numpy backend)
    np.random.seed(seed)
    responses = np.random.poisson(responses_clean, responses_clean.shape)

    return {"centers": centers, "latents": latents, "responses": responses,
            "responses_clean": responses_clean}

def convert_to_angles(points):
    """Transform points on T2 to angles.
    Parameters
    ----------
    points : array-like, shape=[..., 2, 2]
        Points on T2.
    Returns
    -------
    angles : array-like, shape=[..., 2]
        Angles corresponding to points on T2.
    """
    angles = gs.stack([gs.arctan2(*points[:, 0].T),
                       gs.arctan2(*points[:, 1].T)], axis=1)
    return angles

def convert_to_points(angles):
    """Transform angles on T2 to points.
    Parameters
    ----------
    angles : array-like, shape=[..., 2]
        Angles corresponding to points on T2.
    Returns
    -------
    points : array-like, shape=[..., 2, 2]
        Points on T2.
    """
    points = gs.stack(
        [gs.stack([gs.sin(angles[:, 0]), gs.cos(angles[:, 0])], axis=1),
         gs.stack([gs.sin(angles[:, 1]), gs.cos(angles[:, 1])], axis=1)],
        axis=1
    )
    return points

# Test angle/vector conversions on T2
t2 = NFoldManifold(Hypersphere(dim=1), n_copies=2)
points = t2.random_point(n_samples=1000)
angles = convert_to_angles(points)
points_ = convert_to_points(angles)
if gs.sum((points - points_ )**2) < 1e-9:
    print("test angle/vector conversions: passed.")
else:
    raise ValueError("test angle/vector conversions: failed.")

def make_population(ensembles, manifolds, num_neurons, num_data, sigmas, scales, global_seed):
    ensembles = ensembles.split("+")
    for e in ensembles:
        assert e in manifolds
    mixed_responses = []
    for i, e in enumerate(ensembles):
        ensemble = generate_data(manifold=manifolds[e], num_neurons=num_neurons,
                                 num_data=num_data, sigma=sigmas[e], scale=scales[e],
                                 seed=global_seed + i * 3982)
        mixed_responses.append(ensemble["responses"].copy())

    return np.concatenate(mixed_responses, axis=0)
