from mpmath import *
import networkx as nx
import matplotlib.pyplot as plt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid

tree = nx.DiGraph()
#tree.add_edges_from([(0, 1, {'weight': 10}), (1, 2, {'weight': 21}), 
                 #   (1, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                 #   (4, 5, {'weight': 54})])

tree.add_edges_from([(0, 1), (0, 2), (1, 3), (0, 4),(0, 5), (3, 6), (4, 7)])

#tree.add_edges_from([(0, 1, {'weight': 10}), (0, 2, {'weight': 21}), 
                 #    (0, 3, {'weight': 31}), (0, 4, {'weight': 40}),
                 #    (0, 5, {'weight': 54})])

class hyp_embed():
    def __init__(self, tree, epsilon, k = 1, is_weighted = False):
        self.tree = tree
        self.k = k
        self.epsilon = epsilon
        self.is_weighted = is_weighted
        self.H2 = Hyperboloid(dim=2, coords_type="extrinsic")
        self.METRIC = self.H2.metric
        
    
    def compute_tau(self):
        
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
        z0 = mu[0] + mu[1]*j
        z = x[0] + x[1]*j
        result = fdiv(fsub(z0, z), fsub(1, fmul(conj(z0), z)))  #transformation function
        x[0] = re(result)
        x[1] = im(result)
        return x

    def add_children(self, p, x, edge_lengths):
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

    def euc_to_hyp_dist(self, x):
        return sqrt(fdiv(fsub(cosh(x), 1), fadd(cosh(x), 1)))

    def embed(self):
        tree = self.tree
        k = self.k
        epsilon = self.epsilon
        is_weighted = self.is_weighted

        coords = zeros(tree.number_of_nodes(), 2)
    
        root_children = list(tree.successors(0))
        d = len(root_children)   
        tau = self.compute_tau()
    
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
            edge_lengths = list(map(self.euc_to_hyp_dist, ones(d, 1) * tau))
        
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

    
    def plot_geodesic_between_two_points(self, initial_point, end_point, n_steps=10, ax=None):
        
        """Plot the geodesic between two points."""
        if not self.H2.belongs(initial_point):
            raise ValueError("The initial point of the geodesic is not in H2.")
        if not self.H2.belongs(end_point):
            raise ValueError("The end point of the geodesic is not in H2.")

        geodesic = self.METRIC.geodesic(initial_point=initial_point, end_point=end_point)

        t = gs.linspace(0.0, 1.0, n_steps)
        points = geodesic(t)
        visualization.plot(points, ax=ax, space="H2_poincare_disk")
    

    def visualize (self):
        embeddings = self.embed()
        geodesics = []
        #print(embeddings)
        for i in range(self.tree.number_of_nodes()):
            x_coord = embeddings[i, 0]
            y_coord = embeddings[i, 1]
            plt.scatter(x_coord, y_coord, label = str(i))
            
            children = list(self.tree.successors(i))
        
            #convert mpmath floats to Python floats
            parent_x = float(nstr(embeddings[i,0], 15))
            parent_y = float(nstr(embeddings[i,1], 15))
            initial_point = gs.array([parent_x, parent_y])
            initial_point = self.H2._ball_to_extrinsic_coordinates(initial_point)
                         
            for child in children:

                child_x = float(nstr(embeddings[child,0], 15))
                child_y = float(nstr(embeddings[child,1], 15))
                end_point =  gs.array([child_x, child_y])
                end_point = self.H2._ball_to_extrinsic_coordinates(end_point)
                
                geodesics.append((initial_point, end_point))
        
        #plot geodesics
        ax = plt.gca()
        for geod in geodesics:
            initial_point, end_point = geod
            self.plot_geodesic_between_two_points(initial_point, end_point, ax=ax)
            
        plt.legend(loc='lower right', ncol=4)
        plt.show()
        
        
     
        return embeddings

test = hyp_embed(tree, 0.5)
result = test.visualize()
print(result)

