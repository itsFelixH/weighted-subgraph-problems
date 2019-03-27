import networkx as nx
import random

import graph_helper as gh
import dynamic_program as dp
import weighted_subgraph_problem as wsp
import ip_generator as ig


def spanning_tree_heuristic(G, mode='max'):
    """Compute weighted subgraph in graph G using a spanning tree heuristic.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (weighted subgraph)
    weight: objective value (weight of H)"""

    T = nx.maximum_spanning_tree(G, weight='weight')
    HT, weightT = dp.solve_dynamic_prog_on_tree(T, mode)

    H = wsp.postprocessing(G, HT, mode)
    weight = gh.weight(H)

    return H, weight


def node_set_heuristic(G, mode='max'):
    """Compute weighted subgraph in graph G using a node set heuristic.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (weighted subgraph)
    weight: objective value (weight of H)"""

    ip = ig.OP()
    R = wsp.preprocessing(G, mode)

    y = ip.add_node_variables(R)
    z = ip.add_edge_variables(R)
    x = ip.add_root_variables(R)
    R_flow = gh.construct_flow_graph(R)
    f = ip.add_flow_variables(R_flow)

    ip.set_wsp_objective(R, mode)

    ip.add_induce_constraints(R)
    ip.add_root_constraints(R)
    ip.add_flow_constraints(R_flow)

    N = R.number_of_nodes()
    bound = 10
    if N <= bound:
        ip.optimize()
        H = wsp.construct_weighted_subgraph(R, ip)
        best_weight = gh.weight(H)
    else:
        H = nx.empty_graph()
        best_weight = 0
        for i in range(10):
            chosen = random.sample(R.nodes(), N-bound)
            constraint_list = []
            for node in chosen:
                constraint_list.append(ip.addConstr(y[node] == 1, name='C'))
            ip.optimize()
            HR = wsp.construct_weighted_subgraph(R, ip)
            HH = wsp.postprocessing(R, HR, mode)
            weight = gh.weight(HH)
            if weight > best_weight:
                best_weight = weight
                H = HH.copy()

            for constraint in constraint_list:
                ip.remove(constraint)

    return H, best_weight







