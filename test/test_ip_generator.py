from gurobipy import *
import networkx as nx
import graph_generator as gg
import graph_helper as gh
import ip_generator as ig


def test_init_op():
    ip = ig.OP()
    
    assert ip
    assert isinstance(ip, Model)
    assert isinstance(ip, ig.OP)
    assert ip.params.outputflag == 0
    assert ip._x == dict()
    assert ip._y == dict()
    assert ip._z == dict()


def test_add_node_variables():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)
    
    y = ip.add_node_variables(G)
    
    assert G.number_of_nodes() == len(y)
    assert G.number_of_nodes() == ip.numVars
    
    for v in G.nodes():
        assert y[v]
        assert ip._y[v]
        assert y[v] == ip._y[v]
    
    for v in ip.getVars():
        assert int(v.varName[1:]) in G.nodes()


def test_add_root_variables():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)

    x = ip.add_root_variables(G)

    assert G.number_of_nodes() == len(x)
    assert G.number_of_nodes() == ip.numVars

    for v in G.nodes():
        assert x[v]
        assert ip._x[v]
        assert x[v] == ip._x[v]

    for v in ip.getVars():
        assert int(v.varName[1:]) in G.nodes()


def test_add_edge_variables():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)
    
    z = ip.add_edge_variables(G)
    
    assert G.number_of_edges() == len(z)
    assert G.number_of_edges() == ip.numVars
    
    for (u, v) in G.edges():
        assert z[u][v]
        assert ip._z[u][v]
        assert z[u][v] == ip._z[u][v]
        
    for v in ip.getVars():
        z = v.varName.split('_')
        assert (int(z[0][1:]), int(z[1])) in G.edges()


def test_add_edge_variables__multigraph():
    ip = ig.OP()
    G, D = gg.random_weighted_spg(100, 50)
    G = nx.convert_node_labels_to_integers(G)

    z = ip.add_edge_variables(G)

    m = 0
    for zu in z.values():
        for zuv in zu.values():
            m += len(zuv)
    assert G.number_of_edges() == m
    assert G.number_of_edges() == ip.numVars

    for (u, v, k) in G.edges(keys=True):
        assert z[u][v][k]
        assert ip._z[u][v][k]
        assert z[u][v][k] == ip._z[u][v][k]

    for v in ip.getVars():
        z = v.varName.split('_')
        assert (int(z[0][1:]), int(z[1]), int(z[2])) in G.edges(keys=True)


def test_add_flow_variables():
    ip = ig.OP()
    G = gg.random_weighted_graph(100, 0.3, 50)
    G_flow = gh.construct_flow_graph(G)

    f = ip.add_flow_variables(G_flow)

    assert 2 * G.number_of_edges() == sum(len(fu) for fu in f.values())
    assert 2 * G.number_of_edges() == ip.numVars

    for (u, v) in G.edges():
        assert f[u][v]
        assert f[v][u]
        assert ip._f[u][v]
        assert ip._f[v][u]
        assert f[u][v] == ip._f[u][v]
        assert f[v][u] == ip._f[v][u]

    for v in ip.getVars():
        f = v.varName.split('_')
        assert (int(f[0][1:]), int(f[1])) in G.edges()


def test_add_flow_variables__edges():
    ip = ig.OP()
    G = gg.random_weighted_graph(100, 0.3, 50)
    G_flow = gh.construct_flow_graph(G)

    f = ip.add_flow_variables(G_flow)
    z = ip.add_edge_variables(G)

    assert 3 * G.number_of_edges() == ip.numVars

    for u in z:
        for v in z[u]:
            assert f[u][v]
            assert f[v][u]


def test_add_flow_variables__multigraph():
    ip = ig.OP()
    G, D = gg.random_weighted_spg(100, 50)
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()
    G_flow = gh.construct_flow_graph(G)

    f = ip.add_flow_variables(G_flow)

    m = 0
    for fu in f.values():
        for fuv in fu.values():
            m += len(fuv)
    assert 2 * G.number_of_edges() == m
    assert 2 * G.number_of_edges() == ip.numVars

    for (u, v, k) in G.edges(keys=True):
        assert f[u][v][k]
        assert f[v][u][k]
        assert ip._f[u][v][k]
        assert ip._f[v][u][k]
        assert f[u][v][k] == ip._f[u][v][k]

    for v in ip.getVars():
        f = v.varName.split('_')
        assert (int(f[0][1:]), int(f[1]), int(f[2])) in G.edges(keys=True)


def test_set_wsp_objective():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)

    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)

    ip.set_wsp_objective(G)
    objective = quicksum(z[u][v] * w for u, v, w in G.edges.data('weight'))\
        - quicksum(y[v] * w for v, w in G.nodes.data('weight'))
    assert ip.getObjective() == objective


def test_set_wsp_objective_multigraph():
    ip = ig.OP()
    G, D = gg.random_weighted_spg(100, 50)
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()

    y = ip.add_node_variables(G)
    z = ip.add_edge_variables(G)

    ip.set_wsp_objective(G)
    objective = quicksum(z[u][v][k] * w for u, v, k, w in G.edges(keys=True, data='weight'))\
        - quicksum(y[v] * w for v, w in G.nodes.data('weight'))
    assert ip.getObjective() == objective


def test_add_induce_constraints():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)

    ip.add_node_variables(G)
    ip.add_edge_variables(G)

    ip.add_induce_constraints(G)
    assert ip.numConstrs == 3 * G.number_of_edges()

    for u, v in G.edges():
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_1')
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_2')
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_3')


def test_add_induce_constraints__multigraph():
    ip = ig.OP()
    G, D = gg.random_weighted_spg(100, 50)
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()

    ip.add_node_variables(G)
    ip.add_edge_variables(G)

    ip.add_induce_constraints(G)
    assert ip.numConstrs == 3 * G.number_of_edges()

    for u, v, k in G.edges(keys=True):
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_'+str(k)+'_1')
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_'+str(k)+'_2')
        assert ip.getConstrByName('I'+str(u)+'_'+str(v)+'_'+str(k)+'_3')


def test_add_root_constraints():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)

    y = ip.add_node_variables(G)
    x = ip.add_root_variables(G)

    ip.add_root_constraints(G)
    assert ip.numConstrs == G.number_of_nodes() + 1

    assert ip.getConstrByName('R_sum')

    for v in G.nodes():
        assert ip.getConstrByName('R'+str(v))


def test_add_flow_constraints():
    ip = ig.OP()
    G = gg.random_weighted_graph(100, 0.1, 50)
    G_flow = gh.construct_flow_graph(G)

    ip.add_node_variables(G)
    ip.add_root_variables(G)
    ip.add_edge_variables(G)
    ip.add_flow_variables(G_flow)

    ip.add_flow_constraints(G, G_flow)
    assert ip.numConstrs == 2 * G.number_of_edges() + G.number_of_nodes()

    for u, v in G.edges():
        assert ip.getConstrByName('F'+str(u)+'_'+str(v)+'_1')
        assert ip.getConstrByName('F'+str(u)+'_'+str(v)+'_2')

    for v in G.nodes():
        assert ip.getConstrByName('F'+str(v))


def test_add_flow_constraints__multigraph():
    ip = ig.OP()
    G, D = gg.random_weighted_spg(100, 50)
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()
    G_flow = gh.construct_flow_graph(G)

    ip.add_node_variables(G)
    ip.add_root_variables(G)
    ip.add_edge_variables(G)
    ip.add_flow_variables(G_flow)

    ip.add_flow_constraints(G, G_flow)
    assert ip.numConstrs == 2 * G.number_of_edges() + G.number_of_nodes()

    for u, v, k in G.edges(keys=True):
        assert ip.getConstrByName('F'+str(u)+'_'+str(v)+'_'+str(k)+'_1')
        assert ip.getConstrByName('F'+str(u)+'_'+str(v)+'_'+str(k)+'_2')

    for v in G.nodes():
        assert ip.getConstrByName('F'+str(v))


def test_op_model():
    m = Model("mip1")
    G = gg.random_weighted_path(100, 50)
    
    x = dict()
    for v in G.nodes():
        x[v] = m.addVar(vtype=GRB.BINARY, name="y"+str(v))
    
    m.setObjective(quicksum(x[v]*w for v, w in G.nodes.data('weight')), GRB.MAXIMIZE)
    m.addConstr(quicksum(x[v] for v in G.nodes()) <= 50)
    m.optimize()
    
    op = ig.OP()
    y = op.add_node_variables(G)
    
    op.setObjective(quicksum(y[v]*w for v, w in G.nodes.data('weight')), GRB.MAXIMIZE)
    op.addConstr(quicksum(y[v] for v in G.nodes()) <= 50)
    op.optimize()
    
    for v in G.nodes():
        assert x[v]
        assert y[v]
        assert x[v] == y[v]
    
    assert x == y
    assert m.objVal == op.objVal
