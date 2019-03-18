import networkx as nx
import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp

min_node_weight = -40
max_node_weight = -1
min_edge_weight = 1
max_edge_weight = 40


def test_solve_on_path__all_subpaths():
    G = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = wsp.solve_on_path__all_subpaths(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_on_path__all_subpaths__min():
    G = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'min'

    (H, objVal) = wsp.solve_on_path__all_subpaths(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_on_tree__all_subtrees():
    G = gg.random_weighted_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = wsp.solve_on_tree__all_subtrees(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_on_tree__all_subtrees__binary():
    G = gg.random_weighted_binary_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = wsp.solve_on_tree__all_subtrees(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_on_tree__all_subtrees__min():
    G1 = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G2 = gg.random_weighted_binary_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'min'

    (H1, objVal1) = wsp.solve_on_tree__all_subtrees(G1, mode)
    (H2, objVal2) = wsp.solve_on_tree__all_subtrees(G2, mode)

    if H1.number_of_nodes() > 0:
        assert nx.is_connected(H1)
    assert objVal1 == gh.weight(H1)

    if H2.number_of_nodes() > 0:
        assert nx.is_connected(H2)
    assert objVal2 == gh.weight(H2)

    for v in H1.nodes():
        assert v in G1.nodes()
    for e in H1.edges():
        assert e in G1.edges()

    for v in H2.nodes():
        assert v in G2.nodes()
    for e in H2.edges():
        assert e in G2.edges()
