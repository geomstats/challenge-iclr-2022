"""
    Code taken from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""

"""Tree traversal util functions."""
from scipy.cluster.hierarchy import to_tree
import networkx as nx

def to_nx_tree(linked):
    tree, nodelist = to_tree(linked, rd=True)
    G = nx.DiGraph()
    for node in nodelist:
        if node.get_left():
            G.add_edge(node.id, node.left.id)
        if node.get_right():
            G.add_edge(node.id, node.right.id)
    return G

def descendants_traversal(tree):
    """Get all descendants non-recursively, in traversal order."""
    n = len(list(tree.nodes()))
    root = n - 1

    traversal = []

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    is_leaf = [len(children[node]) == 0 for node in range(n)]
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            assert node == stack.pop()
            if is_leaf[node]:
                traversal.append(node)

    return traversal[::-1]


def descendants_count(tree):
    """For every node, count its number of descendant leaves, and the number of leaves before it."""
    n = len(list(tree.nodes()))
    root = n - 1

    left = [0] * n
    desc = [0] * n
    leaf_idx = 0

    children = [list(tree.neighbors(node))[::-1] for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = 1
                left[node] = leaf_idx
                leaf_idx += 1
            else:
                desc[node] = sum([desc[c] for c in children_])
                left[node] = left[children_[0]]
            assert node == stack.pop()

    return desc, left
