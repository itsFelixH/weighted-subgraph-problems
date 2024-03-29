import networkx as nx
import numpy as np
import random
import graph_helper as gh
import graph_generator as gg


min_node_weight = -50
max_node_weight = -1
min_edge_weight = 1
max_edge_weight = 50
weights = (-10, 10, -10, 10)


def test_merge_nodes():
    G1 = nx.Graph()
    G1.add_edge(1, 2)

    G2 = nx.Graph()
    G2.add_edge(3, 4)

    G = nx.union(G1, G2)
    
    assert not G.is_directed()
    assert not G.is_multigraph()
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 2

    G = gh.merge_nodes(G, [2, 3], 'merged')
    
    assert not G.is_directed()
    assert not G.is_multigraph()
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_node('merged')
    assert G.has_node(1)
    assert G.has_node(4)


def test_merge_nodes__multigraph():
    G = nx.MultiDiGraph()
    G.add_edges_from([(1, 2), (3, 4), (1, 3), (2, 5)])
    G.add_edge(1, 2)

    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5
    assert G.is_directed()
    assert G.is_multigraph()
    
    G = gh.merge_nodes(G, [2, 3], 'merged')
    
    assert G.is_directed()
    assert G.is_multigraph()
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 5
    assert G.has_node('merged')
    assert G.has_node(1)
    assert G.has_node(4)
    assert G.has_node(5)


def test_merge_nodes__weights():
    (G_old, dic) = gg.random_weighted_grid(50, 50, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G = G_old.copy()
    
    selected_nodes = random.sample(G.nodes(), np.random.randint(2, G.number_of_nodes() / 2))
    
    G = gh.merge_nodes(G, selected_nodes, 'merged')
    
    assert not G.is_directed()    
    assert G.number_of_nodes() == 2500 - len(selected_nodes) + 1
    assert G.has_node('merged')
    
    for node in selected_nodes:
        assert not G.has_node(node)
        for successor in G_old.neighbors(node):
            if G.has_node(successor):
                assert G.has_edge(successor, 'merged')
                assert G.has_edge('merged', successor)
                weights = []
                for edge in G['merged'][successor]:
                    weights.append(G['merged'][successor][edge]['weight'])
                assert G_old[node][successor]['weight'] in weights


def test_merge_nodes__random():
    G_old = gg.random_weighted_graph(100, 200, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G = G_old.copy()

    selected_nodes = random.sample(G.nodes(), np.random.randint(2, G.number_of_nodes() / 2))
    successors = dict()
    for node in selected_nodes:
        successors[node] = G.neighbors(node)
    G = gh.merge_nodes(G, selected_nodes, 'merged')

    assert G.number_of_nodes() == 100 - len(selected_nodes) + 1
    assert G.has_node('merged')
    
    for node in selected_nodes:
        assert not G.has_node(node)
        for successor in G_old.neighbors(node):
            if G.has_node(successor):
                assert G.has_edge('merged', successor)
                weights = []
                for edge in G['merged'][successor]:
                    weights.append(G['merged'][successor][edge]['weight'])
                assert G_old[node][successor]['weight'] in weights


def test_merge_edges_between_u_and_v():
    G = nx.MultiGraph()
    G.add_edges_from([(1, 2), (3, 4), (1, 3), (2, 5)])
    G.add_edge(1, 2)

    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5
    assert G.number_of_edges(1, 2) == 2
    assert G.is_multigraph()

    G = gh.merge_edges_between_u_and_v(G, 1, 2)

    assert G.is_multigraph()
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    assert G.number_of_edges(1, 2) == 1


def test_merge_edges_between_u_and_v__weighted():
    G = nx.MultiGraph()
    G.add_weighted_edges_from([(1, 2, 5), (3, 4, 4.5), (1, 3, 2.7), (2, 5, 1.4)])
    G.add_edge(1, 2, 3.2)

    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5
    assert G.number_of_edges(1, 2) == 2
    assert G.is_multigraph()

    G = gh.merge_edges_between_u_and_v(G, 1, 2)

    assert G.is_multigraph()
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    assert G.number_of_edges(1, 2) == 1


def test_spanning_tree():
    G = gg.random_connected_graph(50, 80, *weights)
    T = gh.spanning_tree(G, 'max')

    T2 = nx.maximum_spanning_tree(G, weight='weight')

    assert nx.is_tree(T)
    assert gh.sum_edge_weights(T) == gh.sum_edge_weights(T2)

def test_spanning_tree__min():
    G = gg.random_connected_graph(50, 80, *weights)
    T = gh.spanning_tree(G, 'min')

    assert nx.is_tree(T)

def test_is_path():
    G1 = gg.random_weighted_path(100, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G2 = gg.random_weighted_graph(20, 30, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G3 = nx.Graph()
    G3.add_nodes_from([1, 7, 4, 'a'])
    G3.add_edges_from([(1, 7), (7, 4), (4, 'a')])
    G4 = gg.ex_graph_path()
    
    assert gh.is_path(G1)
    assert gh.is_path(G3)
    assert gh.is_path(G4)
    assert not gh.is_path(G2)


def test_direct_tree():
    T = gg.random_weighted_tree(50, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    D = gh.direct_tree(T)
    
    assert nx.is_tree(D)
    assert T.number_of_nodes() == D.number_of_nodes()
    assert T.number_of_edges() == D.number_of_edges()

    roots = [d for (v, d) in D.in_degree() if d == 0]
    assert len(roots) == 1
    
    for (v, w) in T.nodes.data('weight'):
        assert D.has_node(v)
        assert D.node[v]['weight'] == w
    for (u, v, w) in T.edges.data('weight'):
        assert D.has_edge(u, v) ^ D.has_edge(v, u)
        if D.has_edge(u, v):
            assert D[u][v]['weight'] == w
        else:
            assert D[v][u]['weight'] == w


def test_direct_tree__root():
    T = gg.random_weighted_tree(50, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    D = gh.direct_tree(T, 20)

    assert nx.is_tree(D)
    assert D.in_degree(20) == 0
    assert T.number_of_nodes() == D.number_of_nodes()
    assert T.number_of_edges() == D.number_of_edges()

    for (v, w) in T.nodes.data('weight'):
        assert D.has_node(v)
        if v != 20:
            assert D.in_degree(v) == 1
        assert D.node[v]['weight'] == w
    for (u, v, w) in T.edges.data('weight'):
        assert D.has_edge(u, v) ^ D.has_edge(v, u)
        if D.has_edge(u, v):
            assert D[u][v]['weight'] == w
        else:
            assert D[v][u]['weight'] == w


def test_construct_flow_graph():
    G = nx.complete_graph(6)
    G_flow = gh.construct_flow_graph(G)

    assert G_flow.is_directed()

    assert G_flow.number_of_nodes() == G.number_of_nodes()
    for v in G.nodes():
        assert v in G_flow.nodes()

    assert G_flow.number_of_edges() == 2 * G.number_of_edges()
    for u, v in G.edges():
        assert G_flow.has_edge(u, v)
        assert G_flow.has_edge(v, u)


def test_construct_flow_graph__multigraph():
    G, D = gg.random_weighted_spg(50, min_node_weight, max_node_weight, min_edge_weight, max_edge_weight)
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()
    G_flow = gh.construct_flow_graph(G)

    assert G_flow.is_directed()

    assert G_flow.number_of_nodes() == G.number_of_nodes()
    for v in G.nodes():
        assert v in G_flow.nodes()

    assert G_flow.number_of_edges() == 2 * G.number_of_edges()
    for u, v, k in G.edges(keys=True):
        assert G_flow.has_edge(u, v, k)
        assert G_flow.has_edge(v, u, k)


def test_weight():
    G = nx.empty_graph()
    G.add_node(0, weight=-3333)
    G.add_node(1, weight=-4)
    G.add_node(2, weight=-2203)
    G.add_node(3, weight=-7830)
    G.add_node(4, weight=0)
    G.add_node(5, weight=-23)
    G.add_node(6, weight=-89)
    G.add_edge(0, 2, weight=35)
    G.add_edge(1, 3, weight=4)
    G.add_edge(2, 5, weight=23)
    G.add_edge(3, 5, weight=78)
    G.add_edge(4, 0, weight=2)

    edges = +35 + 4 + 23 + 78 + 2
    nodes = 3333 + 4 + 2203 + 7830 + 23 + 89
    assert gh.weight(G) == (edges - nodes)


def test_sum_node_weights():
    G = nx.empty_graph()
    G.add_node(0, weight=3333)
    G.add_node(1, weight=4)
    G.add_node(2, weight=-2203)
    G.add_node(3, weight=7830)
    G.add_node(4, weight=0)

    s = 3333 + 4 - 2203 + 7830
    assert gh.sum_node_weights(G) == s


def test_sum_node_weights__random():
    G = nx.empty_graph()
    weight1 = random.randint(1, 40)
    weight2 = random.randint(1, 40)
    weight3 = random.randint(1, 40)
    weight4 = random.randint(1, 40)
    weight5 = random.randint(1, 40)

    G.add_node(0, weight=weight1)
    G.add_node(1, weight=weight2)
    G.add_node(2, weight=weight3)
    G.add_node(3, weight=weight4)
    G.add_node(4, weight=weight5)

    s = weight1 + weight2 + weight3 + weight4 + weight5
    assert gh.sum_node_weights(G) == s


def test_sum_edge_weights():
    G = nx.empty_graph()
    G.add_edge(0, 2, weight=35)
    G.add_edge(1, 3, weight=4)
    G.add_edge(2, 5, weight=23)
    G.add_edge(3, 5, weight=78)
    G.add_edge(4, 0, weight=2)

    s = 35 + 4 + 23 + 78 + 2
    assert gh.sum_edge_weights(G) == s


def test_level_order_list():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    
    order_list = gh.level_order_list(G, 0)
    order_list1 = gh.level_order_list(G, 1)
    order_list2 = gh.level_order_list(G, 2)
    
    assert order_list == [0, 1, 2, 3, 4, 5, 6]
    assert order_list1 == [1, 3, 4]
    assert order_list2 == [2, 5, 6]
    assert gh.level_order_list(G, 3) == [3]
    assert gh.level_order_list(G, 4) == [4]
    assert gh.level_order_list(G, 5) == [5]
    assert gh.level_order_list(G, 6) == [6]


def test_level_list():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

    level1 = gh.level_list(G, 0, 1)
    level2 = gh.level_list(G, 0, 2)
    level3 = gh.level_list(G, 0, 3)

    assert level1 == [0]
    assert level2 == [1, 2]
    assert level3 == [3, 4, 5, 6]


def test_height():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    
    assert gh.height(G, 0) == 3
    assert gh.height(G, 1) == 2
    assert gh.height(G, 2) == 2
    assert gh.height(G, 3) == 1
    assert gh.height(G, 4) == 1
    assert gh.height(G, 5) == 1
    assert gh.height(G, 6) == 1
