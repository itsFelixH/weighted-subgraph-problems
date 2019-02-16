import networkx as nx
from gurobipy import *

import graph_helper as gh
import graph_generator as gg
import ip_generator as ig
import weighted_subgraph_problem as wsp

    
def test_compare_path_results__random():
    G = gg.random_weighted_path(10, 40)
    mode = 'max'
    
    (H1, objValIpRoot) = wsp.solve_full_ip__rooted(G, mode, 0)
    (H2, objValIp) = wsp.solve_full_ip(G, mode, 0)
    (H3, objValIpPath) = wsp.solve_ip_on_path(G, mode, 0)
    (H4, objValIterate) = wsp.solve_on_path__all_subpaths(G, mode, 0)
    (H5, objValDynamic) = wsp.solve_dynamic_prog_on_path(G, mode, 0)
    
    assert objValIpRoot == objValIp
    assert objValIp == objValIpPath
    assert objValIpPath == objValIterate
    assert objValIterate == objValDynamic