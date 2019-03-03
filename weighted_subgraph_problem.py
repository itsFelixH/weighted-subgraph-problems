from itertools import chain, combinations
import networkx as nx
import numpy as np
from gurobipy import *
import graph_helper as gh
import ip_generator as ig
import decomposition_tree as dt


def setup_ip(G, mode='max', root=None, induced=True, flow=False):
    """Setup IP for WSP on graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'
    root : root node for solution
    induced : bool for induced solution
    flow : bool for using flow formulation

    Returns:
    ip : integer program"""

    ip = ig.OP()

    # Create variables
    ip.add_node_variables(G)
    ip.add_edge_variables(G)
    if not root:
        ip.add_root_variables(G)
    if flow:
        G_flow = gh.construct_flow_graph(G)
        ip.add_flow_variables(G_flow)

    # Set objective function
    ip.set_wsp_objective(G, mode)

    # Add constraints
    if induced:
        ip.add_induce_constraints(G)
    if not root:
        ip.add_root_constraints(G)
        if flow:
            ip.add_flow_constraints(G_flow)
    else:
        ip.add_root_constraints(G, root)
        if flow:
            ip.add_flow_constraints(G_flow, root)

    return ip


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


def solve_rooted_ip(G, root, mode='max'):
    """Compute opt weighted subgraph in graph G containing root.
    Parameters:
    G : NetworkX graph
    root : root node for solution
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph containing root)
    weight: objective value (weight of H)"""
    
    ip = setup_ip(G, mode=mode, root=root)
    ip.add_connectivity_constraints(G, root=root)

    ip.optimize()

    H = construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    return H, weight


def solve_rooted_flow_ip(G, root, mode='max'):
    """Compute opt weighted subgraph in graph G containing root.
    Parameters:
    G : NetworkX graph
    root : root node for solution
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph containing root)
    weight: objective value (weight of H)"""

    ip = setup_ip(G, mode=mode, root=root, flow=True)

    ip.optimize()

    H = construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    return H, weight


def solve_full_ip__rooted(G, mode='max'):
    """Compute opt weighted subgraph in graph G using multiple IPs.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""
    
    H = nx.empty_graph()
    weight = 0
    
    for v in G.nodes():
        (H1, objVal) = solve_rooted_ip(G, v, mode)
        
        if mode == 'max':
            if objVal > weight:
                H = H1
                weight = objVal
            elif objVal == weight:
                if H1.number_of_nodes() < H.number_of_nodes():
                    H = H1
                    weight = objVal

        elif mode == 'min':
            if objVal < weight:
                H = H1
                weight = objVal
            elif objVal == weight:
                if H1.number_of_nodes() < H.number_of_nodes():
                    H = H1
                    weight = objVal

    return H, weight


def solve_full_ip(G, mode='max'):
    """Compute opt weighted subgraph in graph G with an IP.
    Parameters:
    G : NetworkX graph

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    objVal: objective value (weight of H)"""
    
    ip = setup_ip(G, mode=mode)
    ip.add_connectivity_constraints(G)
    ip.optimize()

    H = construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    if (mode == 'max' and weight < 0) or (mode == 'min' and weight > 0):
        H = nx.empty_graph()
        weight = 0

    return H, weight


def solve_separation_ip(G, mode='max'):
    """Compute opt weighted subgraph in graph G by separating the connectivity constraints.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""

    ip = setup_ip(G, mode=mode)
    connected = False

    i = 0
    H = nx.empty_graph()
    while not connected:
        ip.optimize()

        H = construct_weighted_subgraph(G, ip)

        if nx.is_connected(H):
            connected = True
        else:
            ip.add_violated_constraint(G, H.nodes())
        i += 1

    weight = ip.objVal

    return H, weight, i


def solve_flow_ip__rooted(G, mode='max'):
    """Compute opt weighted subgraph in graph G using multiple flow IPs.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""

    H = nx.empty_graph()
    weight = 0

    for v in G.nodes():
        (H1, objVal) = solve_rooted_flow_ip(G, v, mode)

        if mode == 'max':
            if objVal > weight:
                H = H1
                weight = objVal
            elif objVal == weight:
                if H1.number_of_nodes() < H.number_of_nodes():
                    H = H1
                    weight = objVal

        elif mode == 'min':
            if objVal < weight:
                H = H1
                weight = objVal
            elif objVal == weight:
                if H1.number_of_nodes() < H.number_of_nodes():
                    H = H1
                    weight = objVal

    return H, weight


def solve_flow_ip(G, mode='max'):
    """Compute opt weighted subgraph in graph G using a flow IP.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""

    ip = setup_ip(G, mode, flow=True)


    ip.optimize()

    H = construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    if (mode == 'max' and weight < 0) or (mode == 'min' and weight > 0):
        H = nx.empty_graph()
        weight = 0

    return H, weight


def solve_ip_on_path(G, mode='max'):
    """Compute opt weighted subgraph on path G using a IP.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""
    
    if not gh.is_path(G):
        print('G is not a path!')
    
    ip = setup_ip(G, mode)
    x = ip.get_x()
    y = ip.get_y()
    z = ip.get_z()

    # Add connectivity constraints
    path = list(G.nodes)    
    subpaths = [[]]
    
    for i in range(len(path) + 1): 
        for j in range(i + 1, len(path) + 1): 
            sub = path[i:j]
            subpaths.append(sub)

    for s in subpaths:
        elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
        for v in s:
            ip.addConstr(y[v] <= (quicksum(x[u] for u in s)) + (quicksum(z[v1][v2] for v1, v2 in elist[:])))
    
    # Solve
    ip.optimize()
    
    # Construct subgraph
    H = construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    if (mode == 'max' and weight < 0) or (mode == 'min' and weight > 0):
        H = nx.empty_graph()
        weight = 0

    return H, weight


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


def solve_dynamic_prog_on_path(G, mode='max'):
    """Compute weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (maximum/minimum weighted subgraph)
    objVal: objective value (weight of H)"""
    
    if not gh.is_path(G):
        print('G is not a path!')
    
    H_in = [0]
    weight_H_in = -G.node[0]['weight']
    
    H_out = []
    weight_H_out = 0

    for (u, v) in G.edges():
        weight1 = weight_H_in + G[u][v]['weight'] - G.node[v]['weight']
        weight2 = -G.node[v]['weight']
        
        if mode == 'max':
            if weight_H_in > weight_H_out:
                H_out = H_in.copy()
                weight_H_out = weight_H_in

            if weight1 > weight2:
                H_in.append(v)
                weight_H_in = weight1
            else:
                H_in = [v]
                weight_H_in = weight2
        elif mode == 'min':
            if weight_H_in < weight_H_out:
                H_out = H_in.copy()
                weight_H_out = weight_H_in

            if weight1 < weight2:
                H_in.append(v)
                weight_H_in = weight1
            else:
                H_in = [v]
                weight_H_in = weight2

    nodelist = []
    weight = 0
    if mode == 'max':
        if weight_H_in > weight_H_out:
            nodelist = H_in.copy()
            weight = weight_H_in
        else:
            nodelist = H_out.copy()
            weight = weight_H_out
    elif mode == 'min':
        if weight_H_in < weight_H_out:
            nodelist = H_in.copy()
            weight = weight_H_in
        else:
            nodelist = H_out.copy()
            weight = weight_H_out
    
    H = G.subgraph(nodelist)
    
    return H, weight


def solve_dynamic_prog_on_tree(G, mode='max'):
    """Compute weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (maximum/minimum weighted subgraph)
    objVal: objective value (weight of H)"""
    
    if not nx.is_tree(G):
        print('G is not a tree!')
    
    if not G.is_directed():
        G = gh.direct_tree(G)
    root = [v for v, d in G.in_degree() if d == 0]
    root = root[0]
    
    h = gh.height(G, root)
    level = dict()
    for i in range(1, h+1):
        level[i] = gh.level_list(G, root, i)
    
    H_in = dict()
    weight_H_in = dict()
    H_out = dict()
    weight_H_out = dict()

    for v in G.nodes():
        H_in[v] = [v]
        weight_H_in[v] = -G.node[v]['weight']
        H_out[v] = []
        weight_H_out[v] = 0

    for i in reversed(range(1, h)):
        for v in level[i]:
            best_weight = 0
            for w in G.successors(v):
                weight_in = weight_H_in[w]
                weight_out = weight_H_out[w]
                if mode == 'max':
                    if weight_in > weight_out and weight_in > best_weight:
                        best_weight = weight_in
                        H_out[v] = H_in[w].copy()
                        weight_H_out[v] = weight_in
                    elif weight_out > best_weight:
                        best_weight = weight_out
                        H_out[v] = H_out[w].copy()
                        weight_H_out[v] = weight_out
                elif mode == 'min':
                    if weight_in < weight_out and weight_in < best_weight:
                        best_weight = weight_in
                        H_out[v] = H_in[w].copy()
                        weight_H_out[v] = weight_in
                    elif weight_out < best_weight:
                        best_weight = weight_out
                        H_out[v] = H_out[w].copy()
                        weight_H_out[v] = weight_out

                if mode == 'max' and weight_in + G[v][w]['weight'] > 0:
                    H_in[v].extend(H_in[w])
                    weight_H_in[v] += weight_in + G[v][w]['weight']
                elif mode == 'min' and weight_in + G[v][w]['weight'] < 0:
                    H_in[v].extend(H_in[w])
                    weight_H_in[v] += weight_in + G[v][w]['weight']
    
    if (mode == 'max' and weight_H_in[root] > weight_H_out[root]) or (mode == 'min' and weight_H_in[root] < weight_H_out[root]):
        H = G.subgraph(H_in[root]).to_undirected()
        weight = weight_H_in[root]
    else:
        H = G.subgraph(H_out[root]).to_undirected()
        weight = weight_H_out[root]
    
    return H, weight


def solve_dynamic_prog_on_spg(G, D, mode='max'):
    """Compute weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph
    D : Decomposition tree of G
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (maximum/minimum weighted subgraph)
    weight: objective value (weight of H)"""

    H_s = dict()
    weight_H_s = dict()
    H_t = dict()
    weight_H_t = dict()
    H_empty= dict()
    weight_H_empty= dict()
    H_stc = dict()
    weight_H_stc = dict()
    H_stn = dict()
    weight_H_stn = dict()

    for tree in D.get_leaves():
        GD = tree.graph
        s = tree.s
        t = tree.t

        H_s[tree] = {s}
        weight_H_s[tree] = - GD.node[s]['weight']
        H_t[tree] = {t}
        weight_H_t[tree] = - GD.node[t]['weight']
        H_empty[tree] = {}
        weight_H_empty[tree] = 0
        H_stc[tree] = {s, t}
        weight_H_stc[tree] = GD[s][t][0]['weight'] - GD.node[s]['weight'] - GD.node[t]['weight']
        H_stn[tree] = {s, t}
        weight_H_stn[tree] = - GD.node[s]['weight'] - GD.node[t]['weight']

    for i in reversed(range(1, D.depth())):
        for tree in D.level_list(i):
            if not dt.DecompositionTree.is_leaf(tree):
                # children
                GD = tree.graph
                D1 = tree.left
                D2 = tree.right
                s = GD.node[tree.s]
                t = GD.node[tree.t]

                if tree.composition == 'P':  # parallel composition
                    # H_s
                    weights = [weight_H_s[D1], weight_H_s[D2], weight_H_s[D1] + weight_H_s[D2] + s['weight']]
                    nodelist = [H_s[D1], H_s[D2], H_s[D1].union(H_s[D2])]

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.s in new_nodelist:
                        new_nodelist.remove(D1.s)
                        new_nodelist.add(tree.s)
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.t)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.s)
                    if D2.t in new_nodelist:
                        new_nodelist.remove(D2.t)
                        new_nodelist.add(tree.t)

                    H_s[tree] = new_nodelist
                    weight_H_s[tree] = weights[ind]

                    # H_t
                    weights = [weight_H_t[D1], weight_H_t[D2], weight_H_t[D1] + weight_H_t[D2] + t['weight']]
                    nodelist = [H_t[D1], H_t[D2], H_t[D1].union(H_t[D2])]

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.s in new_nodelist:
                        new_nodelist.remove(D1.s)
                        new_nodelist.add(tree.s)
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.t)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.s)
                    if D2.t in new_nodelist:
                        new_nodelist.remove(D2.t)
                        new_nodelist.add(tree.t)

                    H_t[tree] = new_nodelist
                    weight_H_t[tree] = weights[ind]

                    # H_empty
                    weights = [weight_H_empty[D1], weight_H_empty[D2]]
                    nodelist = [H_empty[D1], H_empty[D2]]

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.s in new_nodelist:
                        new_nodelist.remove(D1.s)
                        new_nodelist.add(tree.s)
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.t)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.s)
                    if D2.t in new_nodelist:
                        new_nodelist.remove(D2.t)
                        new_nodelist.add(tree.t)

                    H_empty[tree] = new_nodelist
                    weight_H_empty[tree] = weights[ind]

                    # H_stc
                    weights = [weight_H_stc[D1], weight_H_stc[D2], weight_H_stc[D1] + weight_H_stc[D2] + s['weight']
                               + t['weight'], weight_H_stc[D1] + weight_H_s[D2] + s['weight'], weight_H_stc[D1] +
                               weight_H_t[D2] + t['weight'], weight_H_s[D1] + weight_H_stc[D2] + s['weight'],
                               weight_H_t[D1] + weight_H_stc[D2] + t['weight'], weight_H_stc[D1] + weight_H_stn[D2]
                               + s['weight'] + t['weight'], weight_H_stn[D1] + weight_H_stc[D2] + s['weight']
                               + t['weight']]
                    nodelist = [H_stc[D1], H_stc[D2], H_stc[D1].union(H_stc[D2]), H_stc[D1].union(H_s[D2]),
                                H_stc[D1].union(H_t[D2]), H_s[D1].union(H_stc[D2]), H_t[D1].union(H_stc[D2]),
                                H_stc[D1].union(H_stn[D2]), H_stn[D1].union(H_stc[D2])]

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.s in new_nodelist:
                        new_nodelist.remove(D1.s)
                        new_nodelist.add(tree.s)
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.t)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.s)
                    if D2.t in new_nodelist:
                        new_nodelist.remove(D2.t)
                        new_nodelist.add(tree.t)

                    H_stc[tree] = new_nodelist
                    weight_H_stc[tree] = weights[ind]

                    # H_stn
                    weights = [weight_H_stn[D1], weight_H_stn[D2], weight_H_stn[D1] + weight_H_stn[D2] + s['weight']
                               + t['weight'], weight_H_stn[D1] + weight_H_s[D2] + s['weight'], weight_H_stn[D1]
                               + weight_H_t[D2] + t['weight'], weight_H_s[D1] + weight_H_stn[D2] + s['weight'],
                               weight_H_t[D1] + weight_H_stn[D2] + t['weight']]
                    nodelist = [H_stn[D1], H_stn[D2], H_stn[D1].union(H_stn[D2]), H_stn[D1].union(H_s[D2]),
                                H_stn[D1].union(H_t[D2]), H_s[D1].union(H_stn[D2]), H_t[D1].union(H_stn[D2])]

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.s in new_nodelist:
                        new_nodelist.remove(D1.s)
                        new_nodelist.add(tree.s)
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.t)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.s)
                    if D2.t in new_nodelist:
                        new_nodelist.remove(D2.t)
                        new_nodelist.add(tree.t)

                    H_stn[tree] = new_nodelist
                    weight_H_stn[tree] = weights[ind]

                else:  # series composition
                    join = GD.node[tree.join]
                    # print('')
                    # print('--------------------------------------------------')
                    # print('NEW NODE')
                    # print('')
                    # print(GD.nodes.data('weight'))
                    # print(GD.edges.data('weight'))
                    # print(tree.join)
                    # print(join)
                    # print(D1.graph.nodes.data('weight'))
                    # print(D1.graph.edges.data('weight'))
                    # print(gh.weight(D1.graph))
                    # print(D2.graph.nodes.data('weight'))
                    # print(D2.graph.edges.data('weight'))
                    # print(gh.weight(D2.graph))

                    # H_s
                    weights = [weight_H_s[D1], weight_H_stc[D1] + weight_H_s[D2] + join['weight']]
                    nodelist = [H_s[D1], H_stc[D1].union(H_s[D2])]
                    # print('')
                    # print('H_s')
                    # print(weights)
                    # print(nodelist)

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.join)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.join)

                    H_s[tree] = new_nodelist
                    weight_H_s[tree] = weights[ind]
                    # print(new_nodelist)
                    # print(weights[ind])

                    # H_t
                    weights = [weight_H_t[D2], weight_H_t[D1] + weight_H_stc[D2] + join['weight']]
                    nodelist = [H_t[D2], H_t[D1].union(H_stc[D2])]
                    # print('')
                    # print('H_t')
                    # print(weights)
                    # print(nodelist)

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.join)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.join)

                    H_t[tree] = new_nodelist
                    weight_H_t[tree] = weights[ind]
                    # print(new_nodelist)
                    # print(weights[ind])

                    # H_empty
                    weights = [weight_H_t[D1] + weight_H_s[D2] + join['weight'], weight_H_empty[D1], weight_H_empty[D2]]
                    nodelist = [H_t[D1].union(H_s[D2]), H_empty[D1], H_empty[D2]]
                    # print('')
                    # print('H_empty')
                    # print(weights)
                    # print(nodelist)

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.join)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.join)

                    H_empty[tree] = new_nodelist
                    weight_H_empty[tree] = weights[ind]
                    # print(new_nodelist)
                    # print(weights[ind])

                    # H_stc
                    new_nodelist = H_stc[D1].union(H_stc[D2])
                    new_nodelist.remove(D1.t)
                    new_nodelist.remove(D2.s)
                    new_nodelist.add(tree.join)
                    H_stc[tree] = new_nodelist
                    weight_H_stc[tree] = weight_H_stc[D1] + weight_H_stc[D2] + join['weight']
                    # print('')
                    # print('H_stc')
                    # print(new_nodelist)
                    # print(weight_H_stc[tree])

                    # H_stn
                    weights = [weight_H_stn[D1] + weight_H_stc[D2] + join['weight'], weight_H_stc[D1] + weight_H_stn[D2]
                               + join['weight'], weight_H_s[D1] + weight_H_t[D2], weight_H_stc[D1] + weight_H_t[D2],
                               weight_H_s[D1] + weight_H_stc[D2]]
                    nodelist = [H_stn[D1].union(H_stc[D2]), H_stc[D1].union(H_stn[D2]), H_s[D1].union(H_t[D2]),
                                H_stc[D1].union(H_t[D2]), H_s[D1].union(H_stc[D2])]
                    # print('')
                    # print('H_stn')
                    # print(weights)
                    # print(nodelist)

                    if mode == 'max':
                        ind = np.argmax(weights)
                    else:
                        ind = np.argmin(weights)

                    new_nodelist = nodelist[ind]
                    if D1.t in new_nodelist:
                        new_nodelist.remove(D1.t)
                        new_nodelist.add(tree.join)
                    if D2.s in new_nodelist:
                        new_nodelist.remove(D2.s)
                        new_nodelist.add(tree.join)

                    H_stn[tree] = new_nodelist
                    weight_H_stn[tree] = weights[ind]
                    # print(new_nodelist)
                    # print(weights[ind])

    # compute solution
    weights = [weight_H_s[D], weight_H_t[D], weight_H_empty[D], weight_H_stc[D]]
    nodelist = [H_s[D], H_t[D], H_empty[D], H_stc[D]]

    if mode == 'max':
        ind = np.argmax(weights)
        weight = weights[ind]
        H = G.subgraph(nodelist[ind])
    else:
        ind = np.argmin(weights)
        weight = weights[ind]
        H = G.subgraph(nodelist[ind])

    return H, weight
