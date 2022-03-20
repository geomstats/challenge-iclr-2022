import networkx as nx
import geomstats.backend as gs
import torch
import numpy as np

class TreeRep():
  """
  Class for running the TreeRep Algorithm.

  The algorithm takes in an n by n symmetric matrix
  with positive entries that should represent a metric.

  It outputs a weighted tree, such that the all pairs
  shortest path metric on the tree is an approximation 
  of the input metric.

  If the input metric is 0-hyperbolic, then we get a 
  tree that exactly represents the input metric. 

  Parameters
  ----------

  d : numpy.ndarray or torch.Tensor
    n by n symmetric matrix whose entreis correspond
    to the input metric. That is d[i,j] is the distance
    between the ith and the jth data point. 
    Required
  
  tol : float
    If any distances in the output tree are smaller than
    tol then we round the distances to 0. 
    Optional, default = 1e-5
  """
  def __init__(self, d, tol = 1e-5):
    self.d = d
    self.n = self.d.shape[0]
    self.S = int(1.3*self.n)
    self.tol = tol
    self.nextroots = list(range(self.n,2*self.n))
    self.nextroots.reverse()
    self.G = nx.Graph()
    self.G.add_nodes_from(range(self.n))
    self.debug = False


  def gid(self,D,w,x,y):
    """
    Inputs
    ---------
    D : n x n numpy.ndarray or torch.Tensor
      That represents the matrix for the metric
    
    w,x,y : ints
      These are indices into the matrix. 
      w is the base
      x and y are two quantites we are taking the
      product of. 
    
    Outputs
    --------
      Returns a float that is the gromov product
      of x with y with respect to base w
    """
    return 0.5*(D[w,x]+D[w,y]-D[x,y])
  
  
  def extract_metric(self):
    """
    Inputs
    ---------
    no inputs
    
    Outputs
    --------
      Returns a numpy.ndarry or torch.Tensor that n x n
      that represents the metric given by the all pairs
      shortest path metrics on the learned tree. 
    """
    d = nx.algorithms.shortest_paths.dense.floyd_warshall(self.G)
    self.D = gs.zeros((self.n,self.n))
    for i in range(self.n):
      for j in range(self.n):
        self.D[i,j] = d[i][j]

    return self.D


  def contract_ra(self, r, a, b, c, V):
    """
    Inputs
    --------
      r : int
        This is a stiener node in the graph
      a : int
        This is a node connected to r.
        We are checking if we should contract
        the edge (r,a)
      b : int
        Some other node connected to r
      c : int
        The final node connected to r
      V : List 
        That is a subset of [x,y,z]. Nodes in this 
        list have had their distanes to r determined 
        and recorded in W. Hence must be zeroed out. 
    
    Outputs
    ----------
      True if the edge is contracted and gives
      the next center node a

      False if the edge is not contracted and
      gived the old center node r
    """
    
    if gs.abs(self.W[r,a]) < self.tol:
      for v in V:
        self.W[r,v] = 0
        self.W[v,r] = 0
      
      if self.G.has_edge(a,r):
        self.G.remove_edge(a,r)
      if self.G.has_edge(b,r):
        self.G.remove_edge(b,r)
      if self.G.has_edge(c,r):
        self.G.remove_edge(c,r)

      self.G.remove_node(r)
      self.nextroots.append(r)

      self.G.add_edge(a,b)
      self.G.add_edge(a,c)

      if self.debug:
        print("Contracting")
        print(r,a,b,c)
        print()

      return True, a
    return False, r


  def sort_into_zones(self, V, r, x, y, z, replaced_root = False):
    """
    Inputs
    ---------
    
    V : list
      Contains the nodes that have yet to be sorted
    r : int
      Steiner node of the universal tree
    x,y,z : int
      Other three nodes of the universal tree. That, is 
      x - r - y
          |
          z
      is the tree. 
    replaced_root : bool
      This stores whether an edge in the above tree has been 
      contracted already. 
  
    Outputs
    ----------
    Returns a 7 elements list of tuples of the form (L,zt,a,b)
    where L is a list of the nodes sorted into that zone
    zt is the zone type
    a is the nodes primarily associated with that zones. For
    zones of type 1, b = a
    b is the node secondarily associated with that zone. For 
    zones of type 2 b is not the same as a
    
    
    This is based on definiton 10 in https://arxiv.org/pdf/2005.03847.pdf
      
    """
    n = self.n
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    Z1 = []
    Z2 = []
    R1 = []

    for w in V:
      a = self.gid(self.W,w,x,y)
      b = self.gid(self.W,w,y,z)
      c = self.gid(self.W,w,x,z)

      if gs.abs(a-b) < self.tol and gs.abs(b-c) < self.tol and gs.abs(c-a) < self.tol:
        if a < self.tol and b < self.tol and c < self.tol and not replaced_root:
          replaced_root = True

          self.W[w,n:] = self.W[r,n:]
          self.W[n:,w] = self.W[n:,r]
          self.W[r,n:] = 0
          self.W[n:,r] = 0

          if self.G.has_edge(x,r):
            self.G.remove_edge(x,r)
          if self.G.has_edge(y,r):
            self.G.remove_edge(y,r)
          if self.G.has_edge(z,r):
            self.G.remove_edge(z,r)

          self.G.remove_node(r)

          self.nextroots.append(r)
          r = w
        
          self.G.add_edge(x,r)
          self.G.add_edge(y,r)
          self.G.add_edge(z,r)
        else:
          R1.append(w)
          self.W[w,r] = (a+b+c)/3
          self.W[r,w] = self.W[w,r]
      elif gs.abs(a-gs.amax([a,b,c])) < 1e-10:
        if gs.abs(self.W[w,z] - b) < self.tol or gs.abs(self.W[w,z] - c) < self.tol:
          Z1.append(w)
        else:
          Z2.append(w)
        self.W[w,r] = a
        self.W[r,w] = self.W[w,r]
      elif gs.abs(b-gs.amax([a,b,c])) < 1e-10:
        if gs.abs(self.W[w,x] - a) < self.tol or gs.abs(self.W[w,x] - c) < self.tol:
          X1.append(w)
        else:
          X2.append(w)
        self.W[w,r] = b
        self.W[r,w] = self.W[w,r]
      elif gs.abs(c-gs.amax([a,b,c])) < 1e-10:
        if gs.abs(self.W[w,y] - b) < self.tol or gs.abs(self.W[w,y] - a) < self.tol:
          Y1.append(w)
        else:
          Y2.append(w)
        self.W[w,r] = c
        self.W[r,w] = self.W[w,r]
      else:
        print(a,b,c)

    Zones = [(R1,1,r,r),(X1,1,x,x),(X2,2,x,r),(Y1,1,y,y),(Y2,2,y,r),(Z1,1,z,z),(Z2,2,z,r)]

    return Zones
  

  def add_steiner_node(self,x,y,z):
    """
    Inputs
    ---------
    
    x,y,z : int
      Points in the metric space
    
    
    Outputs
    ---------
    
    Adds the appropriate steiner node r to create the graph
    x - r - y
        |
        z
    Here the edge lengths are set so that that they respect
    the original metric. This method will contract an edge 
    if the edge length is 0. 
    
    This is based on Lemma 1 and definiton 9 in https://arxiv.org/pdf/2005.03847.pdf
  
    """
    
    r = self.nextroots.pop(-1)

    #check if we need to make W bigger
    if r >= self.S:
      new_s = int(1.3*self.S)
      new_w = gs.zeros((new_s,new_s))
      new_w[:self.S,:self.S] = self.W
      self.S = new_s
      self.W = new_w

    self.G.add_node(r)
    self.G.add_edge(x,r)
    self.G.add_edge(y,r)
    self.G.add_edge(z,r)

    if self.debug:
      print("standard stiener")
      print(r,x,y,z)
      print()

    self.W[r,x] = self.gid(self.W,x,y,z)
    self.W[x,r] = self.W[r,x]
    replaced_root, r = self.contract_ra(r,x,y,z,[x])

    self.W[r,y] = self.gid(self.W,y,x,z)
    self.W[y,r] = self.W[r,y]

    if not replaced_root:
      replaced_root, r = self.contract_ra(r,y,x,z,[x, y])

    self.W[r,z] = self.gid(self.W,z,x,y)
    self.W[z,r] = self.W[r,z]

    if not replaced_root:
      replaced_root, r = self.contract_ra(r,z,x,y,[x, y, z])

    return replaced_root, r

  

  def zone1_helper(self,V,x):
    """
    Inputs
    ---------
    
    x: int
      Primary node associated with the zone
    V : list
      Nodes in the zone
    
    
    Outputs
    ---------
    Does the recurvesive step of the algorithm. 
    
    Specifically, if V is empty of has 1 node it returns
    after connecting that one node to z. 
    
    If the length of V is bigger than 1. It picks 2 others
    nodes y,z from V at random and constructs universal tree
    for x,y,z. 
    
    It then sorts the rest of V into the new zones
    """
  
    if len(V) == 0:
      return []
    
    if len(V) == 1:
      self.G.add_edge(x,V[0])
      if self.debug:
        print("Zone 1")
        print(x,V[0])
        print()
      return []
    
    p = torch.randperm(len(V))

    y = V[p[0]]
    z = V[p[1]]

    V_rem = []
    for i in range(2,len(V)):
      V_rem.append(V[p[i]])

    replaced_root, r = self.add_steiner_node(x,y,z)
    Zones = self.sort_into_zones(V_rem,r,x,y,z,replaced_root)
    return Zones
  
  def zone2_helper(self,V,x,y):
    """
    Inputs
    ---------
    
    x : int
      Primary node associated with the zone
    y : int
      Secondary node associated with the zone
    V : list
      Nodes in the zone
    
    
    Outputs
    ---------
    Does the recurvesive step of the algorithm. 
    
    Specifically, if V is empty of has 1 node it returns
    after connecting that one node to z. 
    
    If the length of V is bigger than 1. It picks 2 others
    nodes y,z from V at random and constructs universal tree
    for x,y,z. 
    
    It then sorts the rest of V into the new zones 
  
    """
    if len(V) == 0:
      return []

    idx = gs.argmin(self.W[y,V]).item()
    p = gs.arange(0,len(V))
    p[0] = idx
    p[idx] = 0

    z = V[p[0]]

    V_rem = []
    for i in range(1,len(V)):
      V_rem.append(V[p[i]])
    if self.G.has_edge(x,y):
      self.G.remove_edge(x,y)
    replaced_root, r = self.add_steiner_node(x,y,z)
    
    Zones = self.sort_into_zones(V_rem,r,x,y,z,replaced_root)
    return Zones
  

  def learn_tree(self):
    """
    Inputs
    ---------
    
    None
    
    
    Outputs
    ---------
    Main mathod that runs the recursive TreeRep Algorithm
    
    self.G has been updated to store the learned
    tree. 
  
    """
    
    # Create the weight matrix for the output tree. 
    # It is bigger than n x n since the algorith
    # adds steiner nodes. 
    self.W = gs.zeros((self.S,self.S))
    self.W[:self.n,:self.n] = self.d

    # Pick the inital 3 poits to starts
    p = torch.randperm(self.n)
    x = p[0].item()
    y = p[1].item()
    z = p[2].item()

    V = []
    for i in range(3,len(p)):
      V.append(p[i].item())

    # Create the universal tree
    replaced_root, r = self.add_steiner_node(x,y,z)

    # sort into zones
    Zones = self.sort_into_zones(V,r,x,y,z,replaced_root)

    # Do the recursion
    while(len(Zones)>0):
      V,zt,a,b = Zones.pop(0)
      if zt == 1:
        new_zones = self.zone1_helper(V,a)
        Zones.extend(new_zones)
      else:
        new_zones = self.zone2_helper(V,a,b)
        Zones.extend(new_zones)

    # Some edge weights may become negative. Set these edge weights to 
    # a small number. 
    for e in self.G.edges():
      self.G[e[0]][e[1]]['weight'] = np.maximum(1e-5,self.W[e[0],e[1]])
    
