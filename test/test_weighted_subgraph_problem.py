import random
import networkx as nx
import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp


def test_solve_rooted_ip__max():
    G1 = gg.random_weighted_tree(10, 40)
    G2 = gg.random_weighted_binary_tree(10, 40)
    G3 = gg.random_weighted_path(10, 40)
    G4, dic = gg.random_weighted_grid(2, 5, 20)
    G5 = gg.random_weighted_graph(10, 0.3, 40)
    root1 = random.choice(list(G1.nodes()))
    root2 = random.choice(list(G2.nodes()))
    root3 = random.choice(list(G3.nodes()))
    root4 = random.choice(list(G4.nodes()))
    root5 = random.choice(list(G5.nodes()))

    mode = 'max'

    (H1, objVal1) = wsp.solve_rooted_ip(G1, root1, mode)
    (H2, objVal2) = wsp.solve_rooted_ip(G2, root2, mode)
    (H3, objVal3) = wsp.solve_rooted_ip(G3, root3, mode)
    (H4, objVal4) = wsp.solve_rooted_ip(G4, root4, mode)
    (H5, objVal5) = wsp.solve_rooted_ip(G5, root5, mode)


    assert nx.is_connected(H1)
    assert objVal1 == gh.weight(H1)
    assert H1.has_node(root1)

    assert nx.is_connected(H2)
    assert objVal2 == gh.weight(H2)
    assert H2.has_node(root2)

    assert nx.is_connected(H3)
    assert objVal3 == gh.weight(H3)
    assert H3.has_node(root3)

    assert nx.is_connected(H4)
    assert objVal4 == gh.weight(H4)
    assert H4.has_node(root4)

    assert nx.is_connected(H5)
    assert objVal5 == gh.weight(H5)
    assert H5.has_node(root5)

    for v in H1.nodes():
        assert v in G1.nodes()
    for e in H1.edges():
        assert e in G1.edges()

    for v in H2.nodes():
        assert v in G2.nodes()
    for e in H2.edges():
        assert e in G2.edges()

    for v in H3.nodes():
        assert v in G3.nodes()
    for e in H3.edges():
        assert e in G3.edges()

    for v in H4.nodes():
        assert v in G4.nodes()
    for e in H4.edges():
        assert e in G4.edges()

    for v in H5.nodes():
        assert v in G5.nodes()
    for e in H5.edges():
        assert e in G5.edges()


def test_solve_full_ip__rooted__max():
    G1 = gg.random_weighted_tree(10, 40)
    G2 = gg.random_weighted_binary_tree(10, 40)
    G3 = gg.random_weighted_path(10, 40)
    G4, dic = gg.random_weighted_grid(2, 5, 20)
    G5 = gg.random_weighted_graph(10, 0.3, 40)

    mode = 'max'

    (H1, objVal1) = wsp.solve_full_ip__rooted(G1, mode)
    (H2, objVal2) = wsp.solve_full_ip__rooted(G2, mode)
    (H3, objVal3) = wsp.solve_full_ip__rooted(G3, mode)
    (H4, objVal4) = wsp.solve_full_ip__rooted(G4, mode)
    (H5, objVal5) = wsp.solve_full_ip__rooted(G5, mode)

    if H1.number_of_nodes() > 0:
        assert nx.is_connected(H1)
    assert objVal1 == gh.weight(H1)

    if H2.number_of_nodes() > 0:
        assert nx.is_connected(H2)
    assert objVal2 == gh.weight(H2)

    if H3.number_of_nodes() > 0:
        assert nx.is_connected(H3)
    assert objVal3 == gh.weight(H3)

    if H4.number_of_nodes() > 0:
        assert nx.is_connected(H4)
    assert objVal4 == gh.weight(H4)

    if H5.number_of_nodes() > 0:
        assert nx.is_connected(H5)
    assert objVal5 == gh.weight(H5)

    for v in H1.nodes():
        assert v in G1.nodes()
    for e in H1.edges():
        assert e in G1.edges()

    for v in H2.nodes():
        assert v in G2.nodes()
    for e in H2.edges():
        assert e in G2.edges()

    for v in H3.nodes():
        assert v in G3.nodes()
    for e in H3.edges():
        assert e in G3.edges()

    for v in H4.nodes():
        assert v in G4.nodes()
    for e in H4.edges():
        assert e in G4.edges()

    for v in H5.nodes():
        assert v in G5.nodes()
    for e in H5.edges():
        assert e in G5.edges()


def test_solve_full_ip__max():
    G1 = gg.random_weighted_tree(10, 40)
    G2 = gg.random_weighted_binary_tree(10, 40)
    G3 = gg.random_weighted_path(10, 40)
    G4, dic = gg.random_weighted_grid(2, 5, 20)
    G5 = gg.random_weighted_graph(10, 0.3, 40)

    mode = 'max'

    (H1, objVal1) = wsp.solve_full_ip(G1, mode)
    (H2, objVal2) = wsp.solve_full_ip(G2, mode)
    (H3, objVal3) = wsp.solve_full_ip(G3, mode)
    (H4, objVal4) = wsp.solve_full_ip(G4, mode)
    (H5, objVal5) = wsp.solve_full_ip(G5, mode)

    if H1.number_of_nodes() > 0:
        assert nx.is_connected(H1)
    assert objVal1 == gh.weight(H1)

    if H2.number_of_nodes() > 0:
        assert nx.is_connected(H2)
    assert objVal2 == gh.weight(H2)

    if H3.number_of_nodes() > 0:
        assert nx.is_connected(H3)
    assert objVal3 == gh.weight(H3)

    if H4.number_of_nodes() > 0:
        assert nx.is_connected(H4)
    assert objVal4 == gh.weight(H4)

    if H5.number_of_nodes() > 0:
        assert nx.is_connected(H5)
    assert objVal5 == gh.weight(H5)

    for v in H1.nodes():
        assert v in G1.nodes()
    for e in H1.edges():
        assert e in G1.edges()

    for v in H2.nodes():
        assert v in G2.nodes()
    for e in H2.edges():
        assert e in G2.edges()

    for v in H3.nodes():
        assert v in G3.nodes()
    for e in H3.edges():
        assert e in G3.edges()

    for v in H4.nodes():
        assert v in G4.nodes()
    for e in H4.edges():
        assert e in G4.edges()

    for v in H5.nodes():
        assert v in G5.nodes()
    for e in H5.edges():
        assert e in G5.edges()

def test_solve_ip_on_path__max():
    G = gg.random_weighted_path(10, 40)
    mode = 'max'

    (H, objVal) = wsp.solve_ip_on_path(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_on_path__all_subpaths__max():
    G = gg.random_weighted_path(15, 40)
    mode = 'max'

    (H, objVal) = wsp.solve_on_path__all_subpaths(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_path__max():
    G = gg.random_weighted_path(15, 40)
    mode = 'max'

    (H, objVal) = wsp.solve_dynamic_prog_on_path(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_path__big():
    G = gg.random_weighted_path(100, 40)
    mode = 'max'

    (H, objVal) = wsp.solve_dynamic_prog_on_path(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


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


def test_compare_tree_results__max():
    G = gg.random_weighted_binary_tree(10, 40)
    mode = 'max'

    (H1, objValIpRoot) = wsp.solve_full_ip__rooted(G, mode)
    (H2, objValIp) = wsp.solve_full_ip(G, mode)
    (H4, objValIterate) = wsp.solve_on_tree__all_subtrees(G, mode)
    (H5, objValDynamic) = wsp.solve_dynamic_prog_on_tree(G, mode)

    assert objValIpRoot == objValIp
    assert objValIp == objValIterate
    assert objValIterate == objValDynamic
