from gurobipy import *

class OP(Model):
  
    def __init__(self, *p, **k):
        super(OP,self).__init__(*p, **k)
        self.setParam('OutputFlag', False)
    
    def add_node_variables(self, nodes, name='y'):
        variables = dict()
        for v in nodes:
            variables[v] = self.addVar(vtype=GRB.BINARY, name=name+str(v))
            
        self.update()
        return variables
    
    def add_edge_variables(self, edges, name='z'):
        variables = dict()
        for u,v in edges:
            if u not in variables:
                variables[u] = dict()
            variables[u][v] = self.addVar(vtype=GRB.BINARY, name=name+str(u)+'_'+str(v))
        
        self.update()
        return variables     