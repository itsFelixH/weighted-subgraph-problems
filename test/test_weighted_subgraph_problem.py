from itertools import combinations
import random
import networkx as nx
import graph_generator as gg
import graph_helper as gh
import weighted_subgraph_problem as wsp

min_node_weight = -40
max_node_weight = -1
min_edge_weight = 1
max_edge_weight = 40
weights = (-10, 10, -10, 10)


def test_isolated_vertices_rule():
    degree_zero_vertices = []

    while len(degree_zero_vertices) == 0:
        G = nx.fast_gnp_random_graph(100, 0.01)
        degree_zero_vertices = [v for (v, d) in G.degree() if d == 0]

    G = wsp.isolated_vertices_rule(G)
    degree_zero_vertices = [v for (v, d) in G.degree() if d == 0]
    assert len(degree_zero_vertices) == 0


def test_parallel_edges_rule():
    G, D = gg.random_weighted_spg(100, *weights)
    G, mapping = wsp.parallel_edges_rule(G)

    for u, v in combinations(G.nodes(), 2):
        assert G.number_of_edges(u, v) <= 1


def test_parallel_edges_rule__weights():
    G = nx.MultiGraph()
    G.add_nodes_from([0, 1, 2])

    weight1 = 0
    neg_weight1 = -11
    for i in range(random.randint(2, 20)):
        weight = random.randint(-10, 10)
        if weight > 0:
            weight1 += weight
        elif weight < 0:
            if weight > neg_weight1:
                neg_weight1 = weight
        G.add_edge(0, 1, weight=weight)

    weight2 = 0
    neg_weight2 = -11
    for i in range(random.randint(2, 20)):
        weight = random.randint(-10, 10)
        if weight > 0:
            weight2 += weight
        elif weight < 0:
            if weight > neg_weight2:
                neg_weight2 = weight
        G.add_edge(1, 2, weight=weight)

    G, mapping = wsp.parallel_edges_rule(G)

    for u, v in combinations(G.nodes(), 2):
        assert G.number_of_edges(u, v) <= 1

    for u, v, w in G.edges(data='weight'):
        if (u == 0 and v == 1) or (u == 1 and v == 0):
            assert w == weight1 or w == neg_weight2

        if (u == 1 and v == 2) or (u == 2 and v == 1):
            assert w == weight2 or w == neg_weight2


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
