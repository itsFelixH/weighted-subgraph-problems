import numpy as np
import graph_helper as gh
import decomposition_tree as dt


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
    weight_H_in = G.node[0]['weight']
    
    H_out = []
    weight_H_out = 0

    for (u, v) in G.edges():
        weight1 = weight_H_in + G[u][v]['weight'] + G.node[v]['weight']
        weight2 = G.node[v]['weight']
        
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
        weight_H_in[v] = G.node[v]['weight']
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
    
    if (mode == 'max' and weight_H_in[root] > weight_H_out[root]) or \
            (mode == 'min' and weight_H_in[root] < weight_H_out[root]):
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
        weight_H_s[tree] = GD.node[s]['weight']
        H_t[tree] = {t}
        weight_H_t[tree] = GD.node[t]['weight']
        H_empty[tree] = {}
        weight_H_empty[tree] = 0
        H_stc[tree] = {s, t}
        weight_H_stc[tree] = GD[s][t][0]['weight'] + GD.node[s]['weight'] + GD.node[t]['weight']
        H_stn[tree] = {s, t}
        weight_H_stn[tree] = GD.node[s]['weight'] + GD.node[t]['weight']

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
                    weights = [weight_H_s[D1], weight_H_s[D2], weight_H_s[D1] + weight_H_s[D2] - s['weight']]
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
                    weights = [weight_H_t[D1], weight_H_t[D2], weight_H_t[D1] + weight_H_t[D2] - t['weight']]
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
                    weights = [weight_H_stc[D1], weight_H_stc[D2], weight_H_stc[D1] + weight_H_stc[D2] - s['weight']
                               - t['weight'], weight_H_stc[D1] + weight_H_s[D2] - s['weight'], weight_H_stc[D1] +
                               weight_H_t[D2] - t['weight'], weight_H_s[D1] + weight_H_stc[D2] - s['weight'],
                               weight_H_t[D1] + weight_H_stc[D2] - t['weight'], weight_H_stc[D1] + weight_H_stn[D2]
                               - s['weight'] - t['weight'], weight_H_stn[D1] + weight_H_stc[D2] - s['weight']
                               - t['weight']]
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
                    weights = [weight_H_stn[D1], weight_H_stn[D2], weight_H_stn[D1] + weight_H_stn[D2] - s['weight']
                               - t['weight'], weight_H_stn[D1] + weight_H_s[D2] - s['weight'], weight_H_stn[D1]
                               + weight_H_t[D2] - t['weight'], weight_H_s[D1] + weight_H_stn[D2] - s['weight'],
                               weight_H_t[D1] + weight_H_stn[D2] - t['weight']]
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

                    # H_s
                    weights = [weight_H_s[D1], weight_H_stc[D1] + weight_H_s[D2] - join['weight']]
                    nodelist = [H_s[D1], H_stc[D1].union(H_s[D2])]

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

                    # H_t
                    weights = [weight_H_t[D2], weight_H_t[D1] + weight_H_stc[D2] - join['weight']]
                    nodelist = [H_t[D2], H_t[D1].union(H_stc[D2])]

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

                    # H_empty
                    weights = [weight_H_t[D1] + weight_H_s[D2] - join['weight'], weight_H_empty[D1], weight_H_empty[D2]]
                    nodelist = [H_t[D1].union(H_s[D2]), H_empty[D1], H_empty[D2]]

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

                    # H_stc
                    new_nodelist = H_stc[D1].union(H_stc[D2])
                    new_nodelist.remove(D1.t)
                    new_nodelist.remove(D2.s)
                    new_nodelist.add(tree.join)
                    H_stc[tree] = new_nodelist
                    weight_H_stc[tree] = weight_H_stc[D1] + weight_H_stc[D2] - join['weight']

                    # H_stn
                    weights = [weight_H_stn[D1] + weight_H_stc[D2] - join['weight'], weight_H_stc[D1] + weight_H_stn[D2]
                               - join['weight'], weight_H_s[D1] + weight_H_t[D2], weight_H_stc[D1] + weight_H_t[D2],
                               weight_H_s[D1] + weight_H_stc[D2]]
                    nodelist = [H_stn[D1].union(H_stc[D2]), H_stc[D1].union(H_stn[D2]), H_s[D1].union(H_t[D2]),
                                H_stc[D1].union(H_t[D2]), H_s[D1].union(H_stc[D2])]

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
