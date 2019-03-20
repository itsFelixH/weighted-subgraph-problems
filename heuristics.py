import graph_helper as gh
import dynamic_program as dp
import weighted_subgraph_problem as wsp


def spanning_tree_heuristic(G, mode='max'):
    """Compute weighted subgraph in graph G using a spanning tree heuristic.
    Parameters:
    G : NetworkX graph
    mode : 'max' or 'min'

    Returns:
    H : NetworkX graph (weighted subgraph)
    weight: objective value (weight of H)"""

    T = gh.spanning_tree(G, mode='max')
    HT, weightT = dp.solve_dynamic_prog_on_tree(T, mode)

    H = wsp.postprocessing(G, HT, mode)
    weight = gh.weight(H)

    return H, weight
