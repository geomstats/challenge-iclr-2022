from gtda.homology import VietorisRipsPersistence
import matplotlib.pyplot as plt
import numpy as np
import geomstats.backend as gs


def compute_persistence(data):
    """Compute VietorisRipsPersistence (up to 2nd homology class) of data. See
    https://giotto-ai.github.io/gtda-docs/latest/modules/generated/homology/gtda.homology.VietorisRipsPersistence.html
    Parameters
    ----------
    data : array-like, shape=[num_points, num_dim]
        Data points (#num_points) in num_dim dimensions.
    Returns
    -------
    diagrams : array-like, shape=[..., 3]
        Simplicial comlexes (birth, death, homology class).
    """
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
    diagrams = VR.fit_transform([data])[0]

    return diagrams

def plot_persistence_diagrams(diagrams, titles=None):
    """Plot persistence diagrams from Giotto's VietorisRipsPersistence.
    Parameters
    ----------
    diagrams : list of: array-like, shape=[..., 3]
        Persistence diagrams.
    """
    num_diagrams = len(diagrams)
    plt.figure(figsize=(4 * num_diagrams, 4))
    for i in range(num_diagrams):
      diagram = diagrams[i]
      plt.subplot(1, num_diagrams, i + 1)
      for j in range(3):
        ind = diagram[:, 2] == j
        plt.scatter(*diagram[ind, :2].T, s=20, label="H_%s" % j)
      limits = (np.min(diagram[:, :2]), np.max(diagram[:, :2]))
      plt.plot(limits, limits, '--', label='')
      plt.grid()
      plt.legend()
      if titles is not None:
        plt.title(titles[i])
      else:
        plt.title("Vietoris Rips Persistent Homology")
      plt.xlabel("Births")
      plt.ylabel("Deaths")
    plt.tight_layout()
    plt.show()

def giotto_objective_function(data, k=2):
  VR = VietorisRipsPersistence(homology_dimensions=[1])
  diagrams = VR.fit_transform([data])[0][:, :2]
  lifetimes = diagrams[:, 1] - diagrams[:, 0]
  sorted = gs.sort(lifetimes)[::-1]
  value = gs.sum(sorted[:k]) - gs.sum(sorted[k:])
  output = [value]
  for i in range(k):
      try:
          output += [sorted[i]]
      except IndexError:
          output += [0.0]
  output += [gs.sum(sorted[k:]), gs.mean(sorted[k:])]

  return output