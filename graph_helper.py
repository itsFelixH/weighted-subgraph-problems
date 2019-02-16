import networkx as nx
import random


def merge_nodes(G, selected_nodes, new_node, node_weight=None):
    """Merges selected_nodes into new_node.
    Parameters:
    G : NetworkX graph
    selected_nodes : list (nodes to merge)
    new_node : (node to merge to)

    Returns:
    G : NetworkX graph"""

    multigraph = 0
    if G.is_multigraph():
        multigraph = 1
    
    if node_weight is not None:
        G.add_node(new_node, weight=node_weight)
    else:
        G.add_node(new_node)
    
    edges = list(G.edges.data('weight'))
    for u, v, weight in edges:
        if multigraph:
            if u in selected_nodes:
                G.add_edge(new_node, v, weight=weight)
            elif v in selected_nodes:
                G.add_edge(u, new_node, weight=weight)
        else:
            if u in selected_nodes:
                if not G.has_edge(new_node, v):
                    G.add_edge(new_node, v, weight=weight)
                else:
                    multigraph = 1
                    if G.is_directed():
                        H = nx.MultiDiGraph()
                    else:
                        H = nx.MultiGraph()
                    H.add_nodes_from(G.nodes())
                    H.add_edges_from(G.edges(data=True))
                    G = H
                    G.add_edge(new_node, v, weight=weight)
            elif v in selected_nodes:
                if not G.has_edge(u, new_node):
                    G.add_edge(u, new_node, weight=weight)
                else:
                    multigraph = 1
                    if G.is_directed():
                        H = nx.MultiDiGraph()
                    else:
                        H = nx.MultiGraph()
                    H.add_nodes_from(G.nodes())
                    H.add_edges_from(G.edges(data=True))
                    G = H
                    G.add_edge(u, new_node, weight=weight)
                    
    for node in selected_nodes:
        G.remove_node(node)

    return G

def createSubgraph(G, node, mode='sucessors'):
    if mode == 'sucessors':
        edges = nx.dfs_successors(G, node)
    elif mode == 'predecessors':
        edges = nx.dfs_predecessors(G, node)
    nodes = []
    for k,v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    return G.subgraph(nodes)

def createSubgraphsBetweenNodes(G, s, t):
    S = []
    for v in G.successors(s):
        paths_between = nx.all_simple_paths(G,source=v,target=t)
        nodes = {node for path in paths_between for node in path}
        nodes.add(s)
        S.append(G.subgraph(nodes))
    return S

def is_path(G):
    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()
    return all([G.has_edge(i, i+1) for i in range(N-1)])

def direct_tree(G, root=None):
    root = random.choice(list(G.nodes))
    
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    
    Q = [root]
    used_nodes = []
    while len(Q) > 0:
        u = Q[0]
        used_nodes.append(u)
        for v in G.neighbors(u):
            if v not in used_nodes:
                H.add_edge(u, v, weight=G[u][v]['weight'])
                Q.append(v)
        Q.remove(u)
    
    return H
    
def weight(G):
    """Computes wsp-weight of a given graph.
    Parameters:
    G : NetworkX graph

    Returns:
    weight : int"""

    weight = sum_edge_weights(G) - sum_node_weights(G)

    return weight

def sum_node_weights(G):
    """Computes sum of node weights of a graph.
    Parameters:
    G : NetworkX graph

    Returns:
    weight : int"""

    weight = 0
    for v, w in G.nodes.data('weight'):
        weight += w

    return weight   


def sum_edge_weights(G):
    """Computes sum of edge weights of a graph.
    Parameters:
    G : NetworkX graph

    Returns:
    weight : int"""
    
    weight = G.size(weight='weight')

    return weight

def level_order_list(G, root):
    h = height(G, root)
    l = []
    for i in range(1, h+1):
        l.extend(level_list(G, root, i))
    return l

def level_list(G, root , level):
    """Returns nodes at a given level"""
    
    if not nx.is_tree(G): 
        return 0
    
    if level == 1: 
        return [root]
    elif level > 1:
        l = []
        for v in G.successors(root):
            l.extend(level_list(G, v, level-1))
        return l

def height(G, root):
    """Computes the height of a tree."""
    
    if not nx.is_tree(G):
        return 0
    
    height1 = 0
    if G.is_directed():
        for v in G.successors(root):
            height2 = height(G, v)
            if height2 > height1:
                height1 = height2
    else:
        for v in G.neighbors(root):
            H = G.copy()
            H.remove_node(root)
            height2 = height(H, v)
            if height2 > height1:
                height1 = height2
    return height1 + 1

def get_edgelist_from_nodelist(nodelist):
    """Computes edgelist from nodelist.
    Parameters:
    nodelist : list (list of connected nodes)

    Returns:
    edgelist : list (list of edges)"""

    edgelist = []
    for u, v in zip(nodelist[:-1], nodelist[1:]):
        edgelist.append((u, v))

    return edgelist


def get_nodelist_from_edgelist(edgelist):
    """Computes nodelist from edgelist.
    Parameters:
    edgelist : list (list of adjacent edges)

    Returns:
    nodelist : list (list of nodes)"""

    nodelist = [edgelist[0][0]]
    for u, v in edgelist:
        nodelist.append(v)

    return nodelist