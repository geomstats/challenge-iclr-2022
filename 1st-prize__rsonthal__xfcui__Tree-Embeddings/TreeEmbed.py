from mpmath import *
import networkx as nx
import matplotlib.pyplot as plt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.poincare_ball import PoincareBall

class HypEmbed():
    """
    This class runs Sarkars algorithm. Sarkars algorithm is as follows:
    
    1) Embed the root node at the origin
    2) Let k be the number of children of the node
    3) Compute cone angles so that we have k cones that are equal and 
       maximal in size
    4) Embed each child in the middle of its cone at the appropriate length
    5) Pick a child node
    6) Perform an isometry that maps the child node to the origin and repeats 
       steps 2 - 5. 
    7) Perform the inverse of the isometry
    8) Repeat for child node that has not been embedded. 
    
    Inputs
    ---------
    
    G : nx.Graph
      This is the tree that we want to embed
    
    epsilon : float
      This is maximally allowed distortion
    
    tau : float (optional)
      This is the scale factor for the metric
    
    k : int (optional)
      This relates to curvature of the manifold. 
    
    is_weighted : bool
      Says if G has edges weights. If false edge weights are assumed to be 
      equal to 0
      
    Outputs
    ----------
    
    self.tree : nx.DiGraph
      This is a directed graph that is rooted at 0
    """
    def __init__(self, G, epsilon, tau = None, k = 1, is_weighted = False):
        edges = nx.edges(G)
        DG = nx.DiGraph()
        DG.add_edges_from(edges)
        self.tree = nx.bfs_tree(G, 0)
        self.tau = tau
        edges = nx.edges(self.tree)
        if is_weighted:
          for e in edges:
            src = e[0]
            dst = e[1]
            self.tree[src][dst]['weight'] = G[src][dst]['weight']
        self.k = k
        self.epsilon = epsilon
        self.is_weighted = is_weighted
        self.H2 = Hyperboloid(dim=2, coords_type="extrinsic")
        self.METRIC = self.H2.metric
        
    
    def compute_tau(self):
        """
        Inputs
        ---------
        
        None
        
        Outputs
        ----------
        
        This computes the scale factor tau based on the given 
        epsilon. This is done using the first algorithm in 
        section 4 in https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf
        """
        tree = self.tree
        k = self.k
        epsilon = self.epsilon
        is_weighted = self.is_weighted
    
        eps = power(10, -10)
        n = tree.number_of_nodes()
        degrees = [tree.degree(i) for i in range(n)]
        d_max = max(degrees)
    
        #cone separation angle beta < pi / d
        beta = fdiv(pi, fmul('1.2', d_max))
        
        #angle for cones 
        alpha = fsub(fdiv(fmul(2, pi), d_max), fmul(2, beta))
        nu = fmul(fmul(fneg(2), k), log(tan(fdiv(beta, 2))))
        
        #Compute for each edge the minimum required length
        #L(v_i, v_j) = -2*k*ln(tan(alpha/2))
        #min length is the same for all edges
        min_length = fmul(fmul(fneg(2), k), log(tan(fdiv(alpha, 2))))
        
        #Compute for each edge the minimum scaling factor
        #eta_vi_vj = L(v_i, v_j) / w(v_i, v_j)
        if is_weighted:
            weights = []
            for v_i, v_j, weight in tree.edges(data=True):
                weights.append(tree[v_i][v_j]['weight'])
            min_weight = min(weights)
            if min_weight == float("inf"):
                min_weight = 1
        else: 
            min_weight = 1
        
        eta_max = fdiv(min_length, min_weight)    
        
        #Select tau > eta_max such that all edges are longer than nu*(1+epsilon)/epsilon
        #min weight of all edges
        tau = fadd(fdiv(fmul(fdiv(nu, min_weight), fadd(1, epsilon)), epsilon), eps)
    
        #if tau <= eta_max, set tau > eta_max
        if tau <= eta_max:
            tau = fadd(eta_max, eps)
        
        return tau

    def hyp_isometry(self, mu, x):
        """
        Inputs
        -------
        
        mu : array like
            This length 2 array is the point that we want our 
            isometry to map to 0
        x : array like
            This length 2 array is the point whose image we want
            to compute under this isometry. 
            
        Outputs
        --------
        
        y : array like
            This length 2 array is the image of x under the isometry
        
        
        Here we think of mu,x,y as living in the C (1 dimensional complex space)
        Then the isometry f(x) is given by
        
        f(x) =.      mu - x
                  -------------
                  1 - conj(mu)x
        where conj(mu) is the complex conjugate of mu. 
        
        Note this is an isometry on the Poincare disk for all choices of mu. 
        """
        z0 = mu[0] + mu[1]*j
        z = x[0] + x[1]*j
        result = fdiv(fsub(z0, z), fsub(1, fmul(conj(z0), z)))  #transformation function
        y = zeros(1,2)
        y[0,0] = re(result)
        y[0,1] = im(result)
        return y[0,:]

    def add_children(self, p, x, edge_lengths):
        """
        Inputs:
        ----------
        
        p : array like
            This is a point in the poincare disk
        
        x : array like
            This is a point in the poincare disk
        
        Note the graph has the edge p - x
        In the directed graph self.tree p is a parent of x
        
        Outputs
        ----------
        
        points0 : array like
            This is a num_children x 2 array that has the coordinates 
            for the children of x in self.tree
        """
        
        #map x to (0, 0)
        p0 = self.hyp_isometry(x, p)
        c = len(edge_lengths)
        q = norm(p0)
        p_angle = acos(fdiv(p0[0], q))
        if p0[1] < 0:
            p_angle = fsub(fmul(2, pi), p_angle)
        
        alpha = fdiv(fmul(2, pi), (c+1))
        points0 = zeros(c, 2)
    
        #place child nodes of x
        for k in range(c):
            angle = fadd(p_angle, fmul(alpha, (k+1)))

            points0[k, 0] = fmul(edge_lengths[k], cos(angle))
            points0[k, 1] = fmul(edge_lengths[k], sin(angle))
        
            #reflect all neighboring nodes by mapping x back to its actual coordinates
            points0[k, :] = self.hyp_isometry(x, points0[k, :])
     
        return points0

    def euc_to_hyp_dist(self, tau):
        """
        Inputs 
        ---------
        
        tau : float
            This is a scale factor 
            
            
        Outputs
        ----------
        
        This is a scale factor, however, if we think of points as
        living in the euclidean disk, then we need to translate this 
        scale factor to one that fits inside the disk. 
        
        That is, let 0 be the origin and x be a point at unit distance 
        (euclidean) from 0. Now tau x (as coordinates) will not lie 
        inside the disk (here the euclidean distance between 0 and tau x 
        is tau). 
        
        Here compute the scalar lambda such that lambda x is in the
        poincare disk and the hyperbolic distance from 0 to lambda x 
        is tau.
        
        """
        return sqrt(fdiv(fsub(cosh(tau), 1), fadd(cosh(tau), 1)))

    def embed(self):
        """
        Inputs
        -------
        None
        
        Outputs
        --------
        
        Runs Sarkars algorithm. See class documentation for a description
        of the algorithm. 
        """
        tree = self.tree
        k = self.k
        epsilon = self.epsilon
        is_weighted = self.is_weighted

        coords = zeros(tree.number_of_nodes(), 2)
    
        root_children = list(tree.successors(0))
        d = len(root_children)   
        if self.tau == None:
          tau = self.compute_tau()
        else:
          tau = self.tau

    
        #lengths of unweighted edges
        edge_lengths = list(map(self.euc_to_hyp_dist, ones(d, 1) * tau))
       
        #lengths of weighted edges
        if is_weighted:
            k = 0
            for child in root_children:
                weight = tree[0][child]['weight']
                edge_lengths[k] = self.euc_to_hyp_dist(fmul(tau, weight))
                k += 1
        # queue containing the nodes whose children we're placing
        q = []
    
        #place children of the root
        for i in range(d):
            coords[root_children[i], 0] = fmul(edge_lengths[i], cos(i * 2 * pi / d))
            coords[root_children[i], 1] = fmul(edge_lengths[i], sin(i * 2 * pi / d))
                        
            q.append(root_children[i])
    
        while len(q) > 0:
            #pop the node whose children we're placing off the queue
            h = q.pop(0)
        
            children = list(tree.successors(h))
            parent = list(tree.predecessors(h))[0]
            num_children = len(children)
        
            for child in children:
                q.append(child)
        
            #lengths of unweighted edges
            edge_lengths = list(map(self.euc_to_hyp_dist, ones(num_children, 1) * tau))
        
            #lengths of weighted edges
            if is_weighted:
                k = 0
                for child in children:
                    weight = tree[h][child]['weight']
                    edge_lengths[k] = self.euc_to_hyp_dist(fmul(tau, weight))
                    k += 1
    
            if num_children > 0:
                R = self.add_children(coords[parent, :], coords[h, :], edge_lengths)
                for i in range(num_children):
                    coords[children[i], :] = R[i, :]
        
        return coords

    
    def plot_geodesic_between_two_points(self, initial_point, end_point, n_steps=12, ax=None):
        """
        Plot the geodesic between initial_point and end_point.
        """
        if not self.H2.belongs(initial_point):
            raise ValueError("The initial point of the geodesic is not in H2.")
        if not self.H2.belongs(end_point):
            raise ValueError("The end point of the geodesic is not in H2.")

        geodesic = self.METRIC.geodesic(initial_point=initial_point, end_point=end_point)

        t = gs.linspace(0.0, 1.0, n_steps)
        points = geodesic(t)[1:-1]
        visualization.plot(points, ax=ax, space="H2_poincare_disk", color = "black", alpha = 0.5)
    

    def visualize(self, ax):
        """
        Plot the embeddings and the geodesics correponding to edges
        in the tree. 
        """
        embeddings = self.embed()
        geodesics = []
        P2 = PoincareBall(2)
        for i in range(self.tree.number_of_nodes()):
            x_coord = embeddings[i, 0]
            y_coord = embeddings[i, 1]
            ax.scatter(x_coord, y_coord, label = str(i), s=100)
            
            children = list(self.tree.successors(i))
        
            #convert mpmath floats to Python floats
            parent_x = float(nstr(embeddings[i,0], 15))
            parent_y = float(nstr(embeddings[i,1], 15))
            

            initial_point = P2.projection(gs.array([parent_x, parent_y]))
            initial_point = self.H2._ball_to_extrinsic_coordinates(initial_point)
                         
            for child in children:

                child_x = float(nstr(embeddings[child,0], 15))
                child_y = float(nstr(embeddings[child,1], 15))
                end_point = P2.projection(gs.array([child_x, child_y]))
                end_point = self.H2._ball_to_extrinsic_coordinates(end_point)
                
                geodesics.append((initial_point, end_point))
        
        #plot geodesics
        for geod in geodesics:
            initial_point, end_point = geod
            self.plot_geodesic_between_two_points(initial_point, end_point, ax=ax)
        
        return embeddings


