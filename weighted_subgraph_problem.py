from itertools import chain, combinations
import networkx as nx
from gurobipy import *
import graph_helper as gh


def construct_weighted_subgraph(G, ip):
    """Construct weighted subgraph from IP.
    Parameters:
    G : NetworkX graph
    ip : integer program (solved)

    Returns:
    H : NetworkX graph"""

    if G.is_multigraph():
        H = nx.MultiGraph()
        for v, w in G.nodes.data('weight'):
            if ip.get_y()[v].x > 0.5:
                H.add_node(v, weight=w)

        for u, v, k, w in G.edges(keys=True, data='weight'):
            if ip.get_z()[u][v][k].x > 0.5:
                H.add_edge(u, v, weight=w)
    else:
        H = nx.empty_graph()
        for v, w in G.nodes.data('weight'):
            if ip.get_y()[v].x > 0.5:
                H.add_node(v, weight=w)

        for u, v, w in G.edges.data('weight'):
            if ip.get_z()[u][v].x > 0.5:
                H.add_edge(u, v, weight=w)

    return H


def preprocessing(G, mode='max'):
    """Preprocessing for WSP on graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    R : NetworkX graph (reduced instance)
    node_mapping : {node} (mapping of removed/replaced edges)
    edge_mapping : {node} (mapping of merged nodes)"""

    if G.is_directed():
        G = G.to_undirected()

    node_mapping = dict()
    edge_mapping = dict()
    R = G.copy()

    while True:
        backup = R.copy()

        # Phase 1
        # Isolated_nodes
        R = isolated_vertices_rule(R)

        # Parallel edges
        R, new_edges = parallel_edges_rule(R, mode)
        edge_mapping.update(new_edges)

        # Adjacent edges
        #R, new_nodes = adjacent_edges_rule(R, mode)
        #node_mapping.update(new_nodes)

        # Chain rule
        #R, new_edges = chain_rule(R, mode)
        #edge_mapping.update(new_edges)

        # Phase 2
        # Mirrored hubs
        #R = mirrored_hubs_rule(R, mode)

        if R.number_of_nodes() == backup.number_of_nodes() and R.number_of_edges() == backup.number_of_edges():
            break

    return R, node_mapping, edge_mapping


def isolated_vertices_rule(G):
    """Removes isolated from the graph G. Part of preprocessing scheme
    Parameters:
    G : NetworkX graph

    Returns:
    R : NetworkX graph (reduced instance)"""

    to_remove = []
    for v in G.nodes():
        if G.degree[v] == 0:
            to_remove.append(v)

    G.remove_nodes_from(to_remove)
    return G


def parallel_edges_rule(G, mode='max'):
    """Applies parallel edges rule. Part of preprocessing scheme
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    R : NetworkX graph (reduced instance)"""

    edge_mapping = dict()

    # Parallel edges
    if G.is_multigraph():
        for u in G.nodes():
            for v in G.nodes():
                num_edges = G.number_of_edges(u, v)
                if G.number_of_edges(u, v) > 1:
                    weight = 0
                    positive = 0
                    negative = 0
                    edge_map = []
                    for edge in G[u][v].copy():
                        if G[u][v][edge]['weight'] >= 0:
                            positive += 1
                            weight += G[u][v][edge]['weight']
                            G.remove_edge(u, v, edge)
                            edge_map.append((u, v, edge))
                        else:
                            negative += 1
                    if G.number_of_edges(u, v) < num_edges:
                        key = G.new_edge_key(u, v)
                        G.add_edge(u, v, key, weight=weight)
                        edge_mapping[(u, v, key)] = edge_map
                    if negative > 0:
                        if positive > 0:
                            for edge in G[u][v].copy():
                                if G[u][v][edge]['weight'] < 0:
                                    G.remove_edge(u, v, edge)
                        else:
                            max_weight = None
                            max_edge = None
                            for edge in G[u][v].copy():
                                if G[u][v][edge]['weight'] < 0:
                                    weight = G[u][v][edge]['weight']
                                    if not max_weight or weight > max_weight:
                                        max_edge = edge
                            for edge in G[u][v].copy():
                                if G[u][v][edge]['weight'] < 0:
                                    if max_edge and edge != max_edge:
                                        G.remove_edge(u, v, edge)
    return G, edge_mapping


def adjacent_edges_rule(G, mode='max'):
    """Applies adjacent edges rule. Part of preprocessing scheme
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    R : NetworkX graph (reduced instance)"""

    node_mapping = dict()

    changed = True
    k = 1
    while changed:
        edges = list(G.edges.data('weight')).copy()
        print(edges)
        changed = False
        for u, v, w in edges:
            if w >= 0 and w + G.node[u]['weight'] >= 0 and w + G.node[v]['weight'] >= 0:
                changed = True
                G = gh.merge_nodes(G, [u, v], 'm' + str(k), w + G.node[u]['weight'] + G.node[v]['weight'])
                node_mapping['m' + str(k)] = [u, v]
                k += 1
                break
    return G, node_mapping


def chain_rule(G, mode='max'):
    """Applies chain rule. Part of preprocessing scheme
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    R : NetworkX graph (reduced instance)"""

    edge_mapping = dict()

    changed = True
    while changed:
        degree_two_nodes = [v for v, d in G.degree() if d == 2]
        target_nodes = [v for v in degree_two_nodes if len(list(G.neighbors(v))) == 2]
        changed = False
        for v in target_nodes:
            if G.node[v]['weight'] <= 0:
                if len(list(G.neighbors(v))) > 1:
                    u, w = G.neighbors(v)
                    for k in G[u][v]:
                        w1 = G[u][v][k]['weight']
                    for k in G[v][w]:
                        w2 = G[v][w][k]['weight']

                    if w1 <= 0 and w2 <= 0:
                        changed = True
                        w3 = G.node[v]['weight']
                        G.remove_node(v)
                        if G.is_multigraph():
                            key = G.new_edge_key(u, w)
                            G.add_edge(u, w, key, weight=w3 + w1 + w2)
                            edge_mapping[(u, w, key)] = [(u, v), (v, w)]
                        else:
                            G.add_edge(u, w, weight=w3 + w1 + w2)
                            edge_mapping[(u, w)] = [(u, v), (v, w)]
                        break
    return G, edge_mapping


def mirrored_hubs_rule(G, mode='max'):
    for u, v in combinations(G.nodes(), 2):
        if G.node[u]['weight'] <= G.node[v]['weight']:
            if G.neighbors(u) == G.neighbors(v):
                all_negative = True
                for w in G.neighbors(u):
                    for edge in G[u][w]:
                        if G[u][w][edge]['weight'] > 0:
                            all_negative = False
                if all_negative:
                    G.remove_node(u)
    return G


def postprocessing(G, H, mode='max'):
    """Preprocessing for WSP on graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph"""

    H = positive_edges_rule(G, H, mode)
    H = negative_nodes_rule(H, mode)
    H = negative_edges_rule(H, mode)
    H = neighboring_nodes_rule(G, H, mode)

    return H


def positive_edges_rule(G, H, mode='max'):
    for (u, v, w) in G.edges(data='weight'):
        if u in H.nodes() and v in H.nodes() and (u,v) not in H.edges():
            if w >= 0:
                H.add_edge(u, v)
    return H


def negative_nodes_rule(H, mode='max'):
    to_remove = []
    temp = H.copy()

    for v, wv in H.nodes(data='weight'):
        weight = wv
        for e in H.edges(v):
            weight += e['weight']
        if weight < 0:
            temp.remove_node(v)
            if nx.is_connected(temp):
                to_remove.append(v)

    H.remove_nodes_from(to_remove)
    return H


def negative_edges_rule(H, mode='max'):
    to_remove = []
    temp = H.copy()

    for e, w in H.edges(data='weight'):
        if w < 0:
            temp.remove_edge(e)
            if nx.is_connected(temp):
                to_remove.append(e)

    H.remove_edges_from(to_remove)
    return H


def neighboring_nodes_rule(G, H, mode='max'):
    nodes_to_add = []

    for v in H.nodes():
        for w in G.neighbors(v):
            if w not in H.nodes() and w not in nodes_to_add:
                weight = G.node[w]['weight']
                for u in G.neighbors(w):
                    if u in H.nodes():
                        if G.is_multigraph():
                            for k in G[w][u]:
                                weight += G[w][u][k]['weight']
                        else:
                            weight += G[w][u]['weight']
                if weight >= 0:
                    nodes_to_add.append(w)

    H.add_nodes_from(nodes_to_add)
    return H


def solve_on_path__all_subpaths(G, mode='max'):
    """Compute weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (maximum/minimum weighted subgraph)
    weight: objective value (weight of H)"""
    
    if not gh.is_path(G):
        print('G is not a path!')
    
    path = list(G.nodes)    
    subpaths = [[]]
    for i in range(len(path) + 1): 
        for j in range(i + 1, len(path) + 1): 
            sub = path[i:j]
            subpaths.append(sub) 
    
    weight = 0
    nodelist = []
    for subpath in subpaths:
        weight1 = gh.weight(G.subgraph(subpath))
        if mode == 'max':
            if weight1 > weight:
                weight = weight1
                nodelist = subpath
            elif weight1 == weight:
                if len(subpath) < len(nodelist):
                    weight = weight1
                    nodelist = subpath
        elif mode == 'min':
            if weight1 < weight:
                weight = weight1
                nodelist = subpath
            elif weight1 == weight:
                if len(subpath) < len(nodelist):
                    weight = weight1
                    nodelist = subpath
    
    H = G.subgraph(nodelist)
    
    return H, weight


def solve_on_tree__all_subtrees(G, mode='max'):
    """Compute weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (maximum/minimum weighted subgraph)
    weight: objective value (weight of H)"""
    
    if not nx.is_tree(G):
        print('G is not a tree!')

    if not G.is_directed():
        G = gh.direct_tree(G)
    root = [v for v, d in G.in_degree() if d == 0]
    root = root[0]

    Q = gh.level_order_list(G, root)[::-1]
    
    tree_map = dict()
    for v in Q:
        tree_map[v] = [[v]]
        for w in G.successors(v):
            for tree in tree_map[w]:
                new_tree = [v]
                new_tree.extend(tree)
                tree_map[v].append(new_tree)
        if G.out_degree(v) == 2:
            (v1, v2) = G.successors(v)
            for tree1 in tree_map[v1]:
                for tree2 in tree_map[v2]:
                    new_tree = [v]
                    new_tree.extend(tree1)
                    new_tree.extend(tree2)
                    tree_map[v].append(new_tree)
        if G.out_degree(v) > 2:
            k = G.out_degree(v)
            successors = list(G.successors(v))
            sets = list(chain.from_iterable(combinations(successors, i) for i in range(2, k+1)))
            for s in sets:
                tree_list = []
                for w in s:
                    tree_list.append(tree_map[w])
                for tree in itertools.product(*tree_list):
                    new_tree = [v]
                    for tree1 in tree:
                        new_tree.extend(tree1)
                    tree_map[v].append(new_tree)

    weight = 0
    nodelist = []
    for v in tree_map:
        for tree in tree_map[v]:
            weight1 = gh.weight(G.subgraph(tree))
            if mode == 'max':
                if weight1 > weight:
                    weight = weight1
                    nodelist = tree
                elif weight1 == weight:
                    if len(tree) < len(nodelist):
                        weight = weight1
                        nodelist = tree
            elif mode == 'min':
                if weight1 < weight:
                    weight = weight1
                    nodelist = tree
                elif weight1 == weight:
                    if len(tree) < len(nodelist):
                        weight = weight1
                        nodelist = tree
    
    H = G.subgraph(nodelist).to_undirected()
    
    return H, weight
