from gurobipy import *


class OP(Model):

    def __init__(self, *p, **k):
        Model.__init__(self, *p, **k)
        self.setParam('OutputFlag', False)
        self._x = dict()
        self._y = dict()
        self._z = dict()
        self._f = dict()

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_z(self):
        return self._z

    def get_f(self):
        return self._f

    def add_node_variables(self, G):
        for v in G.nodes():
            self._y[v] = self.addVar(vtype=GRB.BINARY, name='y' + str(v))

        self.update()
        return self._y

    def add_root_variables(self, G):
        for v in G.nodes():
            self._x[v] = self.addVar(vtype=GRB.BINARY, name='x' + str(v))

        self.update()
        return self._x

    def add_edge_variables(self, G):
        for u, v in G.edges():
            if u not in self._z:
                self._z[u] = dict()
            self._z[u][v] = self.addVar(vtype=GRB.BINARY, name='z' + str(u) + '_' + str(v))
        
        self.update()
        return self._z

    def add_flow_variables(self, G_f):
        for u, v in G_f.edges():
            if u not in self._f:
                self._f[u] = dict()
            self._f[u][v] = self.addVar(vtype=GRB.INTEGER, name='f' + str(u) + '_' + str(v))
        self.update()
        return self._f

    def set_wsp_objective(self, G, mode='max'):
        # Set objective
        if mode == 'max':
            self.setObjective((quicksum(self._z[u][v] * w for u, v, w in G.edges.data('weight')))
                              - (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
        elif mode == 'min':
            self.setObjective((quicksum(self._z[u][v] * w for u, v, w in G.edges.data('weight')))
                              - (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)

    def add_induce_constraints(self, G):
        for u, v in G.edges():
            self.addConstr(self._z[u][v] >= self._y[u] + self._y[v] - 1)
            self.addConstr(self._z[u][v] <= self._y[v])
            self.addConstr(self._z[u][v] <= self._y[u])

    def add_root_constraint(self, G):
        self.addConstr((quicksum(self._x[v] for v in G.nodes())) == 1)

        for v in G.nodes():
            self.addConstr(self._x[v] <= self._y[v])

    def add_violated_constraint(self, G, s):
        elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]

        for v in s:
            self.addConstr(self._y[v] <= quicksum(self._z[u][v] for u, v in elist))
