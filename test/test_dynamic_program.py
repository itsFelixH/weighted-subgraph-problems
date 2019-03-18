import random
import networkx as nx
import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp
import dynamic_program as dp

min_node_weight = -40
max_node_weight = -1
min_edge_weight = 1
max_edge_weight = 40


def test_solve_dynamic_prog_on_path():
    G = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = dp.solve_dynamic_prog_on_path(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_path__min():
    G = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'min'

    (H, objVal) = dp.solve_dynamic_prog_on_path(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_path__big():
    G1 = gg.random_weighted_path(100, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G2 = gg.random_weighted_path(80, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)

    (H1, objVal1) = dp.solve_dynamic_prog_on_path(G1, 'max')
    (H2, objVal2) = dp.solve_dynamic_prog_on_path(G2, 'max')

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


def test_solve_dynamic_prog_on_tree():
    G = gg.random_weighted_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = dp.solve_dynamic_prog_on_tree(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_tree__binary():
    G = gg.random_weighted_binary_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = dp.solve_dynamic_prog_on_tree(G, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H)
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


def test_solve_dynamic_prog_on_tree__min():
    G1 = gg.random_weighted_path(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G2 = gg.random_weighted_binary_tree(15, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'min'

    (H1, objVal1) = dp.solve_dynamic_prog_on_tree(G1, mode)
    (H2, objVal2) = dp.solve_dynamic_prog_on_tree(G2, mode)

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


def test_solve_dynamic_prog_on_tree__big():
    G1 = gg.random_weighted_tree(100, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G2 = gg.random_weighted_binary_tree(100, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)

    (H1, objVal1) = dp.solve_dynamic_prog_on_tree(G1, 'max')
    (H2, objVal2) = dp.solve_dynamic_prog_on_tree(G2, 'max')

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


def test_solve_dynamic_prog_on_spg():
    G, D = gg.random_weighted_spg(20, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = dp.solve_dynamic_prog_on_spg(G, D, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H.to_undirected())
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()


# def test_solve_dynamic_prog_on_spg__min():
#     G, D = gg.random_weighted_spg(20, 30)
#     mode = 'min'
#
#     (H, objVal) = dp.solve_dynamic_prog_on_spg(G, D, mode)
#
#     if H.number_of_nodes() > 0:
#         assert nx.is_connected(H.to_undirected())
#     assert objVal == gh.weight(H)
#
#     for v in H.nodes():
#         assert v in G.nodes()
#     for e in H.edges():
#         assert e in G.edges()
# TODO algorithm returns induced subgraph!


def test_solve_dynamic_prog_on_spg__big():
    G, D = gg.random_weighted_spg(100, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    mode = 'max'

    (H, objVal) = dp.solve_dynamic_prog_on_spg(G, D, mode)

    if H.number_of_nodes() > 0:
        assert nx.is_connected(H.to_undirected())
    assert objVal == gh.weight(H)

    for v in H.nodes():
        assert v in G.nodes()
    for e in H.edges():
        assert e in G.edges()
