from TreeRepresentation import TreeRep
import embedding_mpmath as EM
import networkx as nx
import torch
from mpmath import *
import geomstats.backend as gs

from geomstats.geometry.symmetric_matrices import SymmetricMatrices 
from geomstats.geometry.matrices import Matrices, MatricesMetric 
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.poincare_ball import PoincareBallMetric
from geomstats.geometry.poincare_ball import PoincareBall


class TreeSpace(SymmetricMatrices):
  def __init__(self, d = None):
    n = d.shape[0]
    super(TreeSpace, self).__init__(n)
    self.d = d
    T = TreeRep(d)
    T.learn_tree()
    self.A = nx.adjacency_matrix(T.G, weight = 'weight')
    self.T = T
    self.coords = None
  
  def distortion(self, pair = "Input - Tree"):
    if pair == "Input - Tree":
      D_new = self.T.extract_metric()
      D_old = self.d
    if pair == "Input - embed":
      D_new = self.get_embedded_metric()/self.tau
      D_old = self.d
    if pair == "Tree - embed":
      D_new = self.get_embedded_metric()/self.tau
      D_old = self.T.extract_metric()

    dist = 0
    for i in range(self.n):
      for j in range(i):
        dist += gs.abs(D_new[i,j]-D_old[i,j])/D_old[i,j]
    return 2*dist/(self.n*(self.n-1))

  def get_embedded_metric(self):
    METRIC = PoincareBallMetric(2)
    P2 = PoincareBall(2)
    embedding = gs.zeros((self.n,2))
    for i in range(self.n):
      x = float(nstr(self.coords[i,0], 15))
      y = float(nstr(self.coords[i,1], 15))
      embedding[i,:] = P2.projection(gs.array([x, y]))

    self.distances = gs.zeros((self.n,self.n))
    for i in range(self.n):
      for j in range(i):
        self.distances[i,j] = METRIC.dist(embedding[i,:], embedding[j,:])
        self.distances[j,i] = self.distances[i,j]

    return self.distances

  def gid(self, w,x,y):
    return 0.5(self.d(w,x)+self.d(w,y)-self.d(x,y))
   
  def belongs(self, point, atol = gs.atol):
    w = 0
    for i in range(self.n):
      for j in range(self.n):
        for k in range(self.n):
          a = self.gid(w,i,j)
          b = self.gid(w,i,k)
          c = self.gid(w,j,k)
          if a <= gs.min([b,c]) and b <= gs.min([a,c]) and c <= gs.min([a,b]):
            return False
    return True
  
  def embed_to_poincare_ball(self, epsilon = 0.5, tau = 0.5):
    self.tau = tau
    self.embd = EM.hyp_embed(self.T.G, epsilon = epsilon, tau = tau, is_weighted = True)
    self.coords = self.embd.embed()
    return self.coords
  
  def visualize(self, ax, epsilon = 0.5, tau = None):
    self.embd = EM.hyp_embed(self.T.G, epsilon, tau = tau, is_weighted = True)
    return self.embd.visualize(ax)



