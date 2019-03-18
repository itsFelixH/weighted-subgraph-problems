import networkx as nx
from gurobipy import *
import graph_helper as gh
import ip_generator as ig
import weighted_subgraph_problem as wsp


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

    H = wsp.construct_weighted_subgraph(G, ip)
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

    H = wsp.construct_weighted_subgraph(G, ip)
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

    H = wsp.construct_weighted_subgraph(G, ip)
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

        H = wsp.construct_weighted_subgraph(G, ip)

        if nx.is_connected(H):
            connected = True
        else:
            ip.add_violated_constraint(G, nx.connected_components(H))
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


def solve_flow_ip(G, mode='max', induced=True):
    """Compute opt weighted subgraph in graph G using a flow IP.
    Parameters:
    G : NetworkX graph
    mode : 'min' or 'max'

    Returns:
    H : NetworkX graph (opt weighted subgraph)
    weight: objective value (weight of H)"""

    ip = setup_ip(G, mode, induced=induced, flow=True)

    ip.optimize()

    H = wsp.construct_weighted_subgraph(G, ip)
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
    H = wsp.construct_weighted_subgraph(G, ip)
    weight = ip.objVal

    if (mode == 'max' and weight < 0) or (mode == 'min' and weight > 0):
        H = nx.empty_graph()
        weight = 0

    return H, weight
