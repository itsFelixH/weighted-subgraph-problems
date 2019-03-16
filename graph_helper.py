import networkx as nx
import random


def merge_nodes(G, selected_nodes, new_node, node_weight=None):
    """Merges selected_nodes into new_node.
    Parameters:
    G : NetworkX graph
    selected_nodes : list (nodes to merge)
    new_node : (node to merge to)
    node_weight : (weight of the new node)

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


def merge_edges_between_u_and_v(G, u, v):
    """Merges edges between u and v into a single edge.
    Parameters:
    G : NetworkX graph
    u : node
    v : node

    Returns:
    G : NetworkX graph"""

    w = 0
    weighted = False
    edges = G[u][v].copy()
    for edge in edges:
        if 'weight' in G[u][v][edge]:
            weighted = True
            w += G[u][v][edge]['weight']
        G.remove_edge(u, v, edge)

    if weighted:
        G.add_edge(u, v, weight=w)
    else:
        G.add_edge(u, v)
    return G


def is_path(G):
    """Checks if the graph is a path.
    Parameters:
    G : NetworkX graph

    Returns:
    Boolean"""

    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()
    return all([G.has_edge(i, i+1) for i in range(N-1)])


def direct_tree(G, root=None):
    """Directs a tree graph.
    Parameters:
    G : NetworkX graph (undirected tree)
    root : node (root of the tree)

    Returns:
    H : NetworkX graph (directed tree)"""

    if not root:
        degree_two_vertices = [v for (v, d) in G.degree() if d >= 2]
        if len(degree_two_vertices) > 0:
            root = random.choice(degree_two_vertices)
        else:
            root = random.choice(list(G.nodes()))
    
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


def construct_flow_graph(G):
    """Replaces each edge by two arcs.
    Parameters:
    G : NetworkX graph (undirected)

    Returns:
    G_flow : NetworkX graph (directed)"""

    G_flow = G.to_directed()
    return G_flow


def weight(G):
    """Computes wsp-weight of a given graph.
    Parameters:
    G : NetworkX graph

    Returns:
    weight : int"""

    weight = sum_edge_weights(G) + sum_node_weights(G)

    return weight


def sum_node_weights(G):
    """Computes sum of node weights of a graph.
    Parameters:
    G : NetworkX graph

    Returns:
    weight : int"""

    weight = 0
    for v, w in G.nodes.data('weight'):
        if w:
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
    """Returns list of nodes in a tree level by level from top to bottom.
    Parameters:
    G : NetworkX graph (directed tree)
    root : node (root of the tree)

    Returns:
    l : [node]"""

    h = height(G, root)
    l = []
    for i in range(1, h+1):
        l.extend(level_list(G, root, i))
    return l


def level_list(G, root , level):
    """Returns list of nodes in a tree at a given level.
    Parameters:
    G : NetworkX graph (directed tree)
    root : node (root of the tree)
    level : int (desired level)

    Returns:
    l : [node]"""
    
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
    """Computes the height of a tree.
    Parameters:
    G : NetworkX graph (undirected/directed tree)
    root : node (root of the tree)

    Returns:
    height : int"""
    
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
