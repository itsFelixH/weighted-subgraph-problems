from itertools import chain, combinations
import networkx as nx
from gurobipy import *
import graph_helper as gh
import ip_generator as ig


def solve_rooted_ip(G, root, mode='max'):
    """Compute maximum weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph

    Returns:
    H : NetworkX graph (maximum weighted subgraph)
    objVal: objective value (weight of H)"""
    
    ip = ig.OP()
    
    # Create variables
    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)
    
    # Set objective      
    if mode == 'max':
        ip.setObjective((quicksum(z[u][v]*w for u, v, w in G.edges.data('weight')))
                        - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
    elif mode == 'min':
        ip.setObjective((quicksum(z[u][v]*w for u, v, w in G.edges.data('weight')))
                        - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)
    
    # Add induce constraints
    for u, v in G.edges():
        ip.addConstr(z[u][v] >= y[u] + y[v] - 1)
        ip.addConstr(z[u][v] <= y[v])
        ip.addConstr(z[u][v] <= y[u])

    # Add connectivity constraints
    nodes = list(G.nodes)
    ip.addConstr(y[root] >= 1)
    
    n = G.number_of_nodes()
    subsets = list(chain.from_iterable(combinations(nodes,i) for i in range(n + 1) if root in nodes))
    
    for s in subsets:
        if root in s:           
            elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
            t = [v for v in G.nodes() if v not in s]

            ip.addConstr((quicksum(z[u][v]*n for u, v in elist)) >= (quicksum(y[v] for v in t)))
    
    # Solve
    ip.optimize()
    
    # Construct subgraph
    H = nx.empty_graph()
    weight = ip.objVal

    for v, w in G.nodes.data('weight'):
        if y[v].x > 0.5:
            H.add_node(v, weight=w)

    for u, v, w in G.edges.data('weight'):
        if z[u][v].x > 0.5:
            H.add_edge(u, v, weight=w)
    
    return H, weight


def solve_full_ip__rooted(G, mode='max'):
    
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
    """Compute maximum weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph

    Returns:
    H : NetworkX graph (maximum weighted subgraph)
    objVal: objective value (weight of H)"""
    
    ip = ig.OP()
    
    # Create variables
    x = ip.add_node_variables(G, 'x')
    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)
    
    # Set objective      
    if mode == 'max':
        ip.setObjective((quicksum(z[u][v]*w for u,v,w in G.edges.data('weight')))
            - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
    elif mode == 'min':
        ip.setObjective((quicksum(z[u][v]*w for u,v,w in G.edges.data('weight')))
            - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)
    
    # Add induce constraints
    for u, v in G.edges():
        ip.addConstr(z[u][v] >= y[u] + y[v] - 1)
        ip.addConstr(z[u][v] <= y[v])
        ip.addConstr(z[u][v] <= y[u])
        
    # Add root constraints
    ip.addConstr((quicksum(x[v] for v in G.nodes())) == 1)
    
    for v in G.nodes():
        ip.addConstr(x[v] <= y[v])

    # Add connectivity constraints    
    n = G.number_of_nodes()
    nodes = list(G.nodes)
    subsets = list(chain.from_iterable(combinations(nodes,i) for i in range(n + 1)))
    
    for v in G.nodes():
        for s in subsets:
            if v in s:
                elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
                
                ip.addConstr(y[v] <= (quicksum(x[u] for u in s)) + (quicksum(z[u][v] for u, v in elist)))
    
    # Solve
    ip.optimize()
    
    # Construct subgraph
    H = nx.empty_graph()
    weight = 0
    
    if ip.objVal > 0:
        weight = ip.objVal
        for v, w in G.nodes.data('weight'):
            if y[v].x > 0.5:
                H.add_node(v, weight=w)
    
        for u, v, w in G.edges.data('weight'):
            if z[u][v].x > 0.5:
                H.add_edge(u, v, weight=w)

    return H, weight


def setup_ip(G, mode='max'):
    ip = ig.OP()

    # Create variables
    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)

    ip.set_wsp_objective(G, mode)
    ip.add_induce_constraints(G)

    return ip


def solve_separation(G, mode='max'):
    ip = setup_ip(G, mode)
    connected = False

    while not connected:
        ip.optimize()

        # Construct subgraph
        H = nx.empty_graph()
        for v, w in G.nodes.data('weight'):
            if ip._y[v].x > 0.5:
                H.add_node(v, weight=w)

        for u, v, w in G.edges.data('weight'):
            if ip._z[u][v].x > 0.5:
                H.add_edge(u, v, weight=w)

        # Check if connected
        if nx.is_connected(H):
            connected = True
        else:
            ip.add_violated_constraint(G, H.nodes())

    weight = ip.objVal

    return H, weight


def solve_ip_on_path(G, mode='max'):
    """Compute maximum weighted subgraph in graph G.
    Parameters:
    G : NetworkX graph

    Returns:
    H : NetworkX graph (maximum weighted subgraph)
    objVal: objective value (weight of H)"""
    
    if not gh.is_path(G):
        print('G is not a path!')
    
    ip = ig.OP()
    
    # Create variables
    x = ip.add_node_variables(G, 'x')
    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)
    
    # Set objective      
    if mode == 'max':
        ip.setObjective((quicksum(z[u][v]*w for u, v, w in G.edges.data('weight')))
                        - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
    elif mode == 'min':
        ip.setObjective((quicksum(z[u][v]*w for u, v, w in G.edges.data('weight')))
                        - (quicksum(y[v]*w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)
    
    # Add induce constraints
    for u, v in G.edges():
        ip.addConstr(z[u][v] >= y[u] + y[v] - 1)
        ip.addConstr(z[u][v] <= y[v])
        ip.addConstr(z[u][v] <= y[u])
        
    # Add root constraints
    ip.addConstr((quicksum(x[v] for v in G.nodes())) == 1)
    
    for v in G.nodes():
        ip.addConstr(x[v] <= y[v])

    # Add connectivity constraints
    path = list(G.nodes)    
    subpaths = [[]]
    n = G.number_of_nodes()
    
    for i in range(len(path) + 1): 
        for j in range(i + 1, len(path) + 1): 
            sub = path[i:j]
            subpaths.append(sub)
    
    for v in G.nodes():
        for s in subpaths:
            if v in s:
                elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
            
                ip.addConstr(y[v] <= (quicksum(x[u] for u in s)) + (quicksum(z[u][v] for u, v in elist)))
    
    # Solve
    ip.optimize()
    
    # Construct subgraph
    H = nx.empty_graph()
    weight = 0
    
    if ip.objVal > 0:
        weight = ip.objVal
        for v, w in G.nodes.data('weight'):
            if y[v].x > 0.5:
                H.add_node(v, weight=w)
    
        for u, v, w in G.edges.data('weight'):
            if z[u][v].x > 0.5:
                H.add_edge(u, v, weight=w)

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
    
    root = [v for v, d in G.in_degree() if d == 0]
    Q = gh.level_order_list(G, root[0])[::-1]
    
    tree_map = dict()
    for v in Q:
        tree_map[v] = [[v]]
        for w in G.successors(v):
            for tree in tree_map[w]:
                new_tree = [v]
                new_tree.extend(tree)
                tree_map[v].append(new_tree)
        if len(list(G.successors(v))) > 1:
            (v1, v2) = G.successors(v)
            for tree1 in tree_map[v1]:
                for tree2 in tree_map[v2]:
                    new_tree = [v]
                    new_tree.extend(tree1)
                    new_tree.extend(tree2)
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
    
    H = G.subgraph(nodelist)
    
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
        if weight_H_in > weight_H_out:
            H_out = H_in.copy()
            weight_H_out = weight_H_in
        
        weight1 = weight_H_in + G[u][v]['weight'] - G.node[v]['weight']
        weight2 = -G.node[v]['weight']
        
        if mode == 'max':
            if weight1 > weight2:
                H_in.append(v)
                weight_H_in = weight1
            else:
                H_in = [v]
                weight_H_in = weight2
        elif mode == 'min':
            if weight1 < weight2:
                H_in.append(v)
                weight_H_in = weight1
            else:
                H_in = [v]
                weight_H_in = weight2

    if mode == 'max':
        if weight_H_in > weight_H_out:
            nodelist = H_in
            weight = weight_H_in
        else:
            nodelist = H_out
            weight = weight_H_out
    elif mode == 'min':
        if weight_H_in < weight_H_out:
            nodelist = H_in
            weight = weight_H_in
        else:
            nodelist = H_out
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
    root = [v for v, d in G.in_degree() if d==0]
    root = root[0]
    
    h = gh.height(G, root)
    level = dict()
    for i in range(1, h+1):
        level[i] = gh.level_list(G, root, i)
    
    H_in = dict()
    weight_H_in = dict()
    H_out = dict()
    weight_H_out = dict()

    for v in level[h]:
        H_in[v] = [v]
        weight_H_in[v] = -G.node[v]['weight']
        H_out[v] = []
        weight_H_out[v] = 0
        
    for i in reversed(range(1, h)):
        for v in level[i]:
            H_in[v] = [v]
            weight_H_in[v] = -G.node[v]['weight']
            successors = G.successors(v)
            max_weight = None
            for w in successors:
                weight = weight_H_in[w]
                if max_weight:
                    if weight > max_weight:
                        max_weight = weight
                        H_out[v] = H_in[w].copy()
                        weight_H_out[v] = weight_H_in[w]
                else:
                    max_weight = weight
                    H_out[v] = H_in[w].copy()
                    weight_H_out[v] = weight_H_in[w]
                if weight + G[v][w]['weight'] > 0:
                    H_in[v].extend(H_in[w])
                    weight_H_in[v] += weight + G[v][w]['weight']
    
    if weight_H_in[root] > weight_H_out[root]:
        H = G.subgraph(H_in[root])
        weight = weight_H_in[root]
    else:
        H = G.subgraph(H_out[root])
        weight = weight_H_out[root]
    
    if weight < 0:
        H = nx.empty_graph()
        weight = 0
    
    return H, weight
