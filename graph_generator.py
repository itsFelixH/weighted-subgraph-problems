import networkx as nx
import random

import graph_helper as gh
from decomposition_tree import DecompositionTree


def weight_edges(G, min_weight, max_weight):
    """Randomly weight edges of given graph
    Parameters:
    min_weight : int (minimum weight for edges)
    max_weight : int (maximum weight for edges)

    Returns:
    G : NetworkX graph"""
    
    if G.is_multigraph():
        for (u,v, k) in G.edges(keys=True):
            G[u][v][k]['weight'] = random.randint(min_weight, max_weight)
    else:
        for (u,v) in G.edges():
            G[u][v]['weight'] = random.randint(min_weight, max_weight)
        
    return G


def weight_nodes(G, min_weight, max_weight):
    """Randomly weight nodes of given graph
    Parameters:
    min_weight : int (minimum weight for nodes)
    max_weight : int (maximum weight for nodes)

    Returns:
    G : NetworkX graph"""

    for v in G.nodes():
        nx.set_node_attributes(G, {v: {'weight': random.randint(min_weight, max_weight)}})
        
    return G


def weight_graph(G, min_node_weight, max_node_weight, min_edge_weight, max_edge_weigth):
    """Randomly weight nodes and edges of given graph
    Parameters:
    min_node_weight : int (minimum weight for nodes)
    max_node_weight : int (maximum weight for nodes)
    min_edge_weight : int (minimum weight for edges)
    max_edge_weight : int (maximum weight for edges)

    Returns:
    G : NetworkX graph"""

    G = weight_edges(G, min_edge_weight, max_edge_weigth)
    G = weight_nodes(G, min_node_weight, max_node_weight)

    return G


def random_weighted_path_coinfilp(p, max_weight):
    """Generates random weighted graph of arbitrary lenght.
    Parameters:
    p : float (probability for expending the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    P : NetworkX graph"""

    P = nx.empty_graph()
    
    P.add_node(0, weight=random.randint(1, max_weight))
    P.add_node(1, weight=random.randint(1, max_weight))
    P.add_edge(0, 1, weight=random.randint(1, max_weight))

    i = 1
    while random.random() < p:
        P.add_node(i + 1, weight=random.randint(1, max_weight))
        P.add_edge(i, i + 1, weight=random.randint(1, max_weight))
        i = i+1
    return P


def random_weighted_path(number_of_nodes, max_weight):
    """Generates random weighted graph.
    Parameters:
    number_of_nodes: int (number of nodes in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    P : NetworkX graph"""

    P = nx.path_graph(number_of_nodes)
    
    for (u,v) in P.edges():
        P[u][v]['weight'] = random.randint(1, max_weight)
    for v in P.nodes():
        nx.set_node_attributes(P, {v: {'weight': random.randint(1, max_weight)}})
    return P


def random_weighted_graph(number_of_nodes, p, max_weight):
    """Generates random weighted graph.
    Parameters:
    number_of_nodes: int (number of nodes in the graph)
    p : float (probability for edges in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    G : NetworkX graph"""

    G = nx.gnp_random_graph(number_of_nodes, p)
    for (u,v) in G.edges():
        G[u][v]['weight'] = random.randint(1, max_weight)
    for v in G.nodes():
        nx.set_node_attributes(G, {v: {'weight': random.randint(1, max_weight)}})
    return G


def random_weighted_binary_tree(number_of_nodes, max_weight):
    """Generates random weighted binary tree graph.
    Parameters:
    number_of_nodes: int (number of nodes in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    T : NetworkX graph"""
    
    T = nx.Graph()
    T.add_node(0, weight=random.randint(1, max_weight))
    
    free_edges = [(0, 1), (0, 2)]
    
    while T.number_of_nodes() < number_of_nodes:
        (u, v) = random.choice(free_edges)
        
        T.add_node(v, weight=random.randint(1, max_weight))
        T.add_edge(u, v, weight=random.randint(1, max_weight))        
    
        free_edges.extend([(v, 2*v+1), (v, 2*v+2)])
        free_edges.remove((u, v))
        
    return T


def random_weighted_binary_tree2(number_of_nodes, max_weight):
    """Generates random weighted binary tree graph.
    Parameters:
    number_of_nodes: int (number of nodes in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    T : NetworkX graph"""
    
    T = nx.Graph()
    T.add_node(0, weight=random.randint(1, max_weight))
    
    current_leaves = [0]
    i = 1
    while T.number_of_nodes() < number_of_nodes:
        changed = 0
        backup = current_leaves.copy()
        for v in current_leaves:
            if random.random() > 0.5:
                T.add_node(i, weight=random.randint(1, max_weight))
                T.add_edge(i, v, weight=random.randint(1, max_weight))
                current_leaves.append(i)
                changed = 1
                i +=1
            if random.random() > 0.5:
                T.add_node(i, weight=random.randint(1, max_weight))
                T.add_edge(i, v, weight=random.randint(1, max_weight))
                current_leaves.append(i)
                changed = 1
                i +=1
            current_leaves.remove(v)
            
#        if not changed:
#            current_leaves = backup.copy()
        
    return T


def random_weighted_tree(number_of_nodes, max_weight):
    """Generates random weighted tree graph.
    Parameters:
    number_of_nodes: int (number of nodes in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    T : NetworkX graph"""
    
    T = nx.random_tree(number_of_nodes)
    
    for (u,v) in T.edges():
        T[u][v]['weight'] = random.randint(1, max_weight)
    for v in T.nodes():
        nx.set_node_attributes(T, {v: {'weight': random.randint(1, max_weight)}})
    return T


def random_weighted_spg(number_of_edges, max_weight):
    """Generates random weighted sp graph.
    Parameters:
    number_of_edges: int (number of edges in the graph)
    max_weight : int (maximum weight for edges)

    Returns:
    SPG : NetworkX graph"""

    sp_list = dict()
    tree_list = dict()
    source = dict()
    sink = dict()
    node_map = dict()

    for i in range(0, number_of_edges):
        sp_list[i] = nx.MultiDiGraph()
        sp_list[i].add_node(str(i)+'_1')
        sp_list[i].add_node(str(i)+'_2')
        sp_list[i].add_edge(str(i)+'_1', str(i)+'_2', weight=random.randint(0, max_weight))
        source[i] = str(i)+'_1'
        sink[i] = str(i)+'_2'
        tree_list[i] = DecompositionTree(graph=sp_list[i], s=str(i)+'_1', t=str(i)+'_2')
    k = 0
    while len(sp_list) > 1:
        key1 = random.choice(list(sp_list.keys()))
        G1 = sp_list[key1]
        del sp_list[key1]
        D1 = tree_list[key1]
        del tree_list[key1]

        key2 = random.choice(list(sp_list.keys()))
        G2 = sp_list[key2]
        D2 = tree_list[key2]

        G = nx.union(G1, G2)
        if random.random() < 0.5:
            combine = [sink[key1], source[key2]]
            G = gh.merge_nodes(G, combine, 's_' + str(k))
            D = DecompositionTree(composition='S', s=D1.s, t=D2.t, join='s_' + str(k), graph=G, left=D1, right=D2)
            D1.set_parent(D)
            D2.set_parent(D)
            node_map['s_' + str(k)] = [sink[key1], source[key2]]
            source[key2] = source[key1]
        else:
            combine1 = [source[key1], source[key2]]
            combine2 = [sink[key1], sink[key2]]
            G = gh.merge_nodes(G, combine1, 'p1_' + str(k))
            G = gh.merge_nodes(G, combine2, 'p2_' + str(k))
            D = DecompositionTree(composition='P', s='p1_'+str(k), t='p2_'+str(k), graph=G, left=D1, right=D2)
            D1.set_parent(D)
            D2.set_parent(D)
            node_map['p1_' + str(k)] = [source[key1], source[key2]]
            node_map['p2_' + str(k)] = [sink[key1], sink[key2]]
            source[key2] = 'p1_' + str(k)
            sink[key2] = 'p2_' + str(k)

        sp_list[key2] = G
        del source[key1]
        del sink[key1]
        tree_list[key2] = D
        k += 1

    key = list(sp_list.keys())[0]
    SPG = sp_list[key].copy()
    DT = tree_list[key]

    SPG = weight_nodes(SPG, 0, max_weight)
    DT.set_graph(SPG)

    for i in range(2, DT.depth() + 1):
        for tree in DT.level_list(i):
            G = tree.parent.graph
            for node in G.nodes():
                if node in tree.graph.nodes():
                    nx.set_node_attributes(tree.graph, {node: {'weight': G.node[node]['weight']}})
                else:
                    if node in node_map:
                        nodes = node_map[node]
                        for v in nodes:
                            if v in tree.graph.nodes():
                                nx.set_node_attributes(tree.graph, {v: {'weight': G.node[node]['weight']}})

    return SPG, DT


def random_weighted_grid(m, n, max_weight):
    """Generates random weighted grid graph.
    Parameters:
    m: int (number of rows of the grid)
    n: int (number of columns of the grid)
    max_weight : int (maximum weight)

    Returns:
    G : NetworkX graph
    dic: dictionary (positions of the nodes)"""

    G = nx.grid_2d_graph(m, n)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, max_weight)
    for v in G.nodes():
        nx.set_node_attributes(G, {v: {'weight': random.randint(1, max_weight)}})
    dic = dict(zip(G, G))
    return G, dic


def ex_graph_path():
    P = nx.empty_graph()
    P.add_node(0, weight=7)
    P.add_node(1, weight=2)
    P.add_node(2, weight=4)
    P.add_node(3, weight=1)
    P.add_node(4, weight=8)
    P.add_node(5, weight=4)
    P.add_node(6, weight=7)
    P.add_node(7, weight=6)
    P.add_node(8, weight=1)
    P.add_node(9, weight=7)
    P.add_node(10, weight=5)
    P.add_node(11, weight=5)
    P.add_node(12, weight=6)
    P.add_node(13, weight=6)
    P.add_node(14, weight=4)    
    
    P.add_edge(0, 1, weight=5)
    P.add_edge(1, 2, weight=10)
    P.add_edge(2, 3, weight=8)
    P.add_edge(3, 4, weight=3)
    P.add_edge(4, 5, weight=6)
    P.add_edge(5, 6, weight=9)
    P.add_edge(6, 7, weight=9)
    P.add_edge(7, 8, weight=2)
    P.add_edge(8, 9, weight=7)
    P.add_edge(9, 10, weight=4)
    P.add_edge(10, 11, weight=3)
    P.add_edge(11, 12, weight=4)
    P.add_edge(12, 13, weight=3)
    P.add_edge(13, 14, weight=9)
    
    return P


def ex_graph_tree():
    T = nx.empty_graph()

    return T
