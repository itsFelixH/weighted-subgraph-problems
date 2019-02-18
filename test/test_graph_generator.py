import networkx as nx
import graph_generator as gg
from decomposition_tree import DecompositionTree


def test_weight_edges():
    G = nx.complete_graph(6)
    G = gg.weight_edges(G, 10, 100)
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 10
        assert data['weight'] <= 100
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']


def test_weight_edges__multigraph():
    G = nx.MultiDiGraph()
    G.add_edges_from([(1, 2), (3, 4), (1, 3), (2, 5)])
    G.add_edge(1, 2)
    G = gg.weight_edges(G, 15, 30)
    
    assert nx.is_directed(G)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5  
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 15
        assert data['weight'] <= 30


def test_weight_edges__directed():
    G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    G = gg.weight_edges(G, 15, 30)
    
    assert nx.is_directed(G)
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 30

    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 15
        assert data['weight'] <= 30


def test_weight_nodes():
    G = nx.complete_graph(6)
    G = gg.weight_nodes(G, 10, 100)
    
    for (v, data) in G.nodes(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 10
        assert data['weight'] <= 100


def test_random_weighted_path_coinfilp():
    G = gg.random_weighted_path_coinfilp(0.9, 50)
    
    assert G.nodes()
    assert G.edges()
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1
    
    for (v, data) in G.nodes(data=True):
        assert G.degree[v] >= 1
        assert G.degree[v] <= 2
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']


def test_random_weighted_path():
    G = gg.random_weighted_path(100, 50)
    
    assert G.nodes()
    assert G.edges()
    assert G.number_of_nodes() == 100
    
    for (v, data) in G.nodes(data=True):
        assert G.degree[v] >= 1
        assert G.degree[v] <= 2
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']


def test_random_weighted_graph():
    G = gg.random_weighted_graph(100, 0.1, 50)
    
    assert G.nodes()
    assert G.edges()
    assert G.number_of_nodes() == 100
    
    for (v, data) in G.nodes(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']


def test_random_weighted_binary_tree():
    G = gg.random_weighted_binary_tree(100, 40)
    
    assert G.nodes()
    assert G.edges()
    assert G.number_of_nodes() == 100
    
    assert nx.is_tree(G)
    assert not nx.is_directed(G)
    
    for (v, data) in G.nodes(data=True):
        assert G.degree[v] >= 1
        assert G.degree[v] <= 3
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 40
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 40


def test_random_weighted_tree():
    G = gg.random_weighted_tree(100, 660)
    
    assert G.nodes()
    assert G.edges()
    assert G.number_of_nodes() == 100
    
    assert nx.is_tree(G)
    
    for (v, data) in G.nodes(data=True):
        assert G.degree[v] >= 1
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 660
        
    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 660
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']


def test_random_weighted_spg():
    (G, D) = gg.random_weighted_spg(100, 50)

    assert G.number_of_edges() == 100
    assert G.nodes()
    assert G.edges()
    
    assert D
    assert isinstance(D, DecompositionTree)
    assert D.size() == 199
    assert len(D.get_leaves()) == 100

    assert nx.is_directed(G)
    assert nx.is_directed_acyclic_graph(G)
    
    for (v, data) in G.nodes(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50

    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50


def test_random_weighted_grid():
    (G, dic) = gg.random_weighted_grid(50, 50, 10)

    assert G.number_of_nodes() == 2500
    assert G.number_of_edges() == 4900
    assert G.nodes()
    assert G.edges()
    
    for (v, data) in G.nodes(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] >= 1
        assert data['weight'] <= 50

    for (u, v, data) in G.edges(data=True):
        assert data['weight']
        assert isinstance(data['weight'], int)
        assert data['weight'] <= 10
        assert G[v][u]
        assert data['weight'] == G[v][u]['weight']
