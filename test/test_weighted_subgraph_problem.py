import networkx as nx
from gurobipy import *

import graph_helper as gh
import graph_generator as gg
import ip_generator as ig
import weighted_subgraph_problem as wsp

    
def test_compare_path_results__max():
    G = gg.random_weighted_path(10, 40)
    mode = 'max'
    
    (H1, objValIpRoot) = wsp.solve_full_ip__rooted(G, mode)
    (H2, objValIp) = wsp.solve_full_ip(G, mode)
    (H3, objValIpPath) = wsp.solve_ip_on_path(G, mode)
    (H4, objValIterate) = wsp.solve_on_path__all_subpaths(G, mode)
    (H5, objValDynamic) = wsp.solve_dynamic_prog_on_path(G, mode)
    
    assert objValIpRoot == objValIp
    assert objValIp == objValIpPath
    assert objValIpPath == objValIterate
    assert objValIterate == objValDynamic