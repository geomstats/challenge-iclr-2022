from TreeRepresentation import TreeRep
import TreeEmbed as EM
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
  """
  This class extends symmetric matrices. This class is to represent
  a single point in the space of all trees on n data points. 
  
  Here a tree will be represented as the combinatorial tree structure
  as well as an n x n distance matrix
  
  Inputs
  --------
  
  d : numpy.ndarray or torch.Tesnor
    d is the input n x n distance matrix
    
  Outputs
  ---------
  
  The constructor for the class takes the metric data and calls TreeRep
  and gets the combinatorial structure
  
  self.A stores the weight matrix for this structure. 
  
  Other quantities that may be initialized by other methods
  
  self.distances : numpy.ndarry or torch.Tensor 
    n x n matrix. When it exists, will store the distances from the 
    learned embeddings
  
  self.T : TreeRep object
    Stores the TreeRep object for this metric data
  
  self.tau : float
    The scale factors for Sarkar's algorithm
  
  self.coords : ndarry or torch.Tensor
    n x 2 array that stores the coordinates in the Poincare disk 
    for the embeddings 
    
  self.embd : HypEmbed object
    Object that performed Sarkar's algorithm for the tree self.T.G
  
  """
  def __init__(self, d = None):
    n = d.shape[0]
    super(TreeSpace, self).__init__(n)
    self.d = d
    T = TreeRep(d)
    T.learn_tree()
    #self.A = nx.adjacency_matrix(T.G, weight = 'weight')
    self.T = T
    self.coords = None
  
  def distortion(self, pair = "Input - Tree"):
    """
    Inputs
    ---------
    pair : string
      Indicates which pair of metrics we should use 
      when computing the average distortion
      
      The options are 
        1) between the input and the tree learned
        2) between the input and the metric from the embeddings
        3) between the metric from the tree and the embedding
    
    Outputs
    ---------
    
    dist : float
      float represrenting the average distortion. 
      
    """
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
    """
    Inputs
    --------
    
    None
    
    Outputs
    ---------
    
    Computes the hyperbolic metric between n points for the 
    calculated embedding self.coords
    
    stores the output in self.distances 
    
    """
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
    """
    computes the gromov inner product for x and y with respect to
    base point w using the input metric. 
    """
    return 0.5(self.d(w,x)+self.d(w,y)-self.d(x,y))
   
  def belongs(self, atol = gs.atol):
    """
    Checks if the given input metric is 0 hyperbolic
    """
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
    """
    Inputs
    ---------
      epsilon : float (optional)
        This is maximum multiplicative distortion that we want 
        between our tree metric and the embedded metric. 
        
      tau : float (optional)
        This is the scale of the metric. 
        If tau is specified, then epsilon is ignored. 
        
    Outputs
    ---------
    
    Computes the embedding into the Poincare disk using Sarkar's
    algorithm.
    
    Stores the embeddings in self.coords.
    """
    self.tau = tau
    self.embd = EM.HypEmbed(self.T.G, epsilon = epsilon, tau = tau, is_weighted = True)
    self.coords = self.embd.embed()
    return self.coords
  
  def visualize(self, ax, epsilon = 0.5, tau = None):
    """
    Inputs
    ---------
    
    ax : axes plot
      This plot on which we plot the embedding
      
    Outputs
    ----------
    
    Plots the embedding, along with the geodesics between data points
    that are connected via edges in the data. 
    """
    self.embd = EM.HypEmbed(self.T.G, epsilon, tau = tau, is_weighted = True)
    return self.embd.visualize(ax)



