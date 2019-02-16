import networkx as nx
from gurobipy import *

import graph_helper as gh
import graph_generator as gg
import ip_generator as ig


def test_init_op():
    ip = ig.OP()
    
    assert ip
    assert isinstance(ip, Model)
    assert isinstance(ip, ig.OP)
    assert ip.params.outputflag == 0

def test_add_node_variables():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)
    
    y = ip.add_node_variables(G.nodes())
    
    assert G.number_of_nodes() == len(y)
    
    for v in G.nodes():
        assert y[v]
    
    for v in ip.getVars():
        assert int(v.varName[1:]) in G.nodes()

def test_add_edge_variables():
    ip = ig.OP()
    G = gg.random_weighted_path(100, 50)
    
    z = ip.add_edge_variables(G.edges())
    
    assert G.number_of_edges() == len(z)
    
    for (u, v) in G.edges():
        assert z[u][v]
        
    for v in ip.getVars():
        z = v.varName.split('_')
        assert (int(z[0][1:]), int(z[1])) in G.edges()

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
    y = op.add_node_variables(G.nodes())
    
    op.setObjective(quicksum(y[v]*w for v, w in G.nodes.data('weight')), GRB.MAXIMIZE)
    op.addConstr(quicksum(y[v] for v in G.nodes()) <= 50)
    op.optimize()
    
    for v in G.nodes():
        assert x[v]
        assert y[v]
        assert x[v] == y[v]
    
    assert x == y
    assert m.objVal == op.objVal
        
    