from gurobipy import *
from itertools import chain, combinations


class OP(Model):

    def __init__(self, *p, **k):
        """Creates an new OP as a child of Gurobi.Model.

        Returns:
        OP"""

        Model.__init__(self, *p, **k)
        self.setParam('OutputFlag', False)
        self._x = dict()
        self._y = dict()
        self._z = dict()
        self._f = dict()

    def get_x(self):
        """Getter for x

        Returns:
        x : dict"""

        return self._x

    def get_y(self):
        """Getter for y

        Returns:
        y : dict"""

        return self._y

    def get_z(self):
        """Getter for z

        Returns:
        z : dict"""

        return self._z

    def get_f(self):
        """Getter for f

        Returns:
        f : dict"""

        return self._f

    def add_node_variables(self, G):
        """Adds a variable to OP for each node in G
        Parameters:
        G : NetworkX Graph

        Returns:
        y : dict"""

        for v in G.nodes():
            self._y[v] = self.addVar(vtype=GRB.BINARY, name='y' + str(v))

        self.update()
        return self._y

    def add_root_variables(self, G):
        """Adds a variable to OP for each node in G for encoding a root node
        Parameters:
        G : NetworkX Graph

        Returns:
        x : dict"""

        for v in G.nodes():
            self._x[v] = self.addVar(vtype=GRB.BINARY, name='x' + str(v))

        self.update()
        return self._x

    def add_edge_variables(self, G):
        """Adds a variable to OP for each edge in G
        Parameters:
        G : NetworkX Graph

        Returns:
        z : dict"""

        if G.is_multigraph():
            for u, v, k in G.edges(keys=True):
                if u not in self._z:
                    self._z[u] = dict()
                if v not in self._z[u]:
                    self._z[u][v] = dict()
                self._z[u][v][k] = self.addVar(vtype=GRB.BINARY, name='z' + str(u) + '_' + str(v) + '_' + str(k))
                self._z[u][v][k].BranchPriority = 9

        else:
            for u, v in G.edges():
                if u not in self._z:
                    self._z[u] = dict()
                self._z[u][v] = self.addVar(vtype=GRB.BINARY, name='z' + str(u) + '_' + str(v))
                self._z[u][v].BranchPriority = 9

        self.update()
        return self._z

    def add_flow_variables(self, G_flow):
        """Adds a variable to OP for each edge in G_f to use as flow variables
        Parameters:
        G_f : NetworkX DiGraph

        Returns:
        y : dict"""

        if G_flow.is_multigraph():
            for u, v, k in G_flow.edges(keys=True):
                if u not in self._f:
                    self._f[u] = dict()
                if v not in self._f[u]:
                    self._f[u][v] = dict()
                self._f[u][v][k] = self.addVar(vtype=GRB.INTEGER, name='f' + str(u) + '_' + str(v) + '_' + str(k))
        else:
            for u, v in G_flow.edges():
                if u not in self._f:
                    self._f[u] = dict()
                self._f[u][v] = self.addVar(vtype=GRB.INTEGER, name='f' + str(u) + '_' + str(v))
        self.update()
        return self._f

    def set_wsp_objective(self, G, mode='max'):
        """Sets the objective function for OP
        Parameters:
        G : NetworkX Graph
        mode : string ('min' or 'max')"""

        if G.is_multigraph():
            if mode == 'max':
                self.setObjective((quicksum(self._z[u][v][k] * w for u, v, k, w in G.edges(keys=True, data='weight')))
                                  + (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
            elif mode == 'min':
                self.setObjective((quicksum(self._z[u][v][k] * w for u, v, k, w in G.edges(keys=True, data='weight')))
                                  + (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)
        else:
            if mode == 'max':
                self.setObjective((quicksum(self._z[u][v] * w for u, v, w in G.edges.data('weight')))
                                  + (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MAXIMIZE)
            elif mode == 'min':
                self.setObjective((quicksum(self._z[u][v] * w for u, v, w in G.edges.data('weight')))
                                  + (quicksum(self._y[v] * w for v, w in G.nodes.data('weight'))), GRB.MINIMIZE)
        self.update()

    def add_induce_constraints(self, G):
        """Adds the constraints for making the subgraph induced
        Parameters:
        G : NetworkX Graph"""

        if G.is_multigraph():
            for u, v, k in G.edges(keys=True):
                self.addConstr(self._z[u][v][k] >= self._y[u] + self._y[v] - 1,
                               name='I'+str(u)+'_'+str(v)+'_'+str(k)+'_1')
                self.addConstr(self._z[u][v][k] <= self._y[v], name='I'+str(u)+'_'+str(v)+'_'+str(k)+'_2')
                self.addConstr(self._z[u][v][k] <= self._y[u], name='I'+str(u)+'_'+str(v)+'_'+str(k)+'_3')
        else:
            for u, v in G.edges():
                self.addConstr(self._z[u][v] >= self._y[u] + self._y[v] - 1, name='I'+str(u)+'_'+str(v)+'_1')
                self.addConstr(self._z[u][v] <= self._y[v], name='I'+str(u)+'_'+str(v)+'_2')
                self.addConstr(self._z[u][v] <= self._y[u], name='I'+str(u)+'_'+str(v)+'_3')
        self.update()

    def add_root_constraints(self, G, root=None):
        """Adds the root constraints
        Parameters:
        G : NetworkX Graph
        root : vertex (root to be included, optional)"""

        if root:
            self.addConstr(self._y[root] == 1, name='R')
        else:
            self.addConstr((quicksum(self._x[v] for v in G.nodes())) == 1, name='R_sum')

            for v in G.nodes():
                self.addConstr(self._x[v] <= self._y[v], name='R'+str(v))
        self.update()

    def add_connectivity_constraints(self, G, root=None):
        """Adds the constraints for making the subgraph connected
        Parameters:
        G : NetworkX Graph
        root : vertex (root, optional)"""

        subsets = chain.from_iterable(combinations(G.nodes, i) for i in range(G.number_of_nodes() + 1))
        if root:
            for s in subsets:
                if root in s:
                    if G.is_multigraph():
                        elist = [e for e in G.edges(keys=True) if (e[0] in s) ^ (e[1] in s)]
                        t = [v for v in G.nodes() if v not in s]
                        self.addConstr((quicksum(self._z[u][v][k] * G.number_of_nodes() for u, v, k in elist[:]))
                                       >= (quicksum(self._y[v] for v in t[:])), name='Conn')
                    else:
                        elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
                        t = [v for v in G.nodes() if v not in s]
                        self.addConstr((quicksum(self._z[u][v] * G.number_of_nodes() for u, v in elist[:]))
                                       >= (quicksum(self._y[v] for v in t[:])), name='Conn')
        else:
            for s in subsets:
                if G.is_multigraph():
                    elist = [e for e in G.edges(keys=True) if (e[0] in s) ^ (e[1] in s)]
                    for v in s:
                        self.addConstr(self._y[v] <= (quicksum(self._x[u] for u in s))
                                       + (quicksum(self._z[v1][v2][k] for v1, v2, k in elist[:])), name='Conn'+str(v))
                else:
                    elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
                    for v in s:
                        self.addConstr(self._y[v] <= (quicksum(self._x[u] for u in s))
                                       + (quicksum(self._z[v1][v2] for v1, v2 in elist[:])), name='Conn'+str(v))
        self.update()

    def add_flow_constraints(self, G_flow, root=None):
        """Adds the constraints for making the subgraph connected using flow
        Parameters:
        G : NetworkX Graph
        G_f : NetworkX DiGraph
        root : vertex (root, optional)"""

        if G_flow.is_multigraph():
            for u, v, k in G_flow.edges(keys=True):
                if u in self._z and v in self._z[u] and k in self._z[u][v]:
                    self.addConstr(self._f[u][v][k] <= G_flow.number_of_nodes() * self._z[u][v][k],
                                   name='F' + str(u) + '_' + str(v) + '_' + str(k) + '_1')
                else:
                    self.addConstr(self._f[u][v][k] <= G_flow.number_of_nodes() * self._z[v][u][k],
                                   name='F' + str(v) + '_' + str(u) + '_' + str(k) + '_2')
            if root:
                arcs = G_flow.edges(keys=True)
                for r in G_flow.nodes():
                    if r != root:
                        incoming = [(u, v, k) for u, v, k in arcs if v == r]
                        outgoing = [(v, w, k) for v, w, k in arcs if v == r]
                        self.addConstr(quicksum(self._f[u][v][k] for u, v, k in incoming[:])
                                       - quicksum(self._f[v][w][k] for v, w, k in outgoing[:])
                                       == self._y[r], name='F' + str(r))
            else:
                arcs = G_flow.edges(keys=True)
                for r in G_flow.nodes():
                    incoming = [(u, v, k) for u, v, k in arcs if v == r]
                    outgoing = [(v, w, k) for v, w, k in arcs if v == r]
                    self.addConstr(quicksum(self._f[u][v][k] for u, v, k in incoming[:])
                                   - quicksum(self._f[v][w][k] for v, w, k in outgoing[:])
                                   >= self._y[r] - self._x[r] * (G_flow.number_of_nodes() + 1), name='F'+str(r))
        else:
            for u, v in G_flow.edges():
                if u in self._z and v in self._z[u]:
                    self.addConstr(self._f[u][v] <= G_flow.number_of_nodes() * self._z[u][v],
                                   name='F'+str(u)+'_'+str(v)+'_1')
                else:
                    self.addConstr(self._f[u][v] <= G_flow.number_of_nodes() * self._z[v][u],
                                   name='F'+str(v)+'_'+str(u)+'_2')

            if root:
                arcs = G_flow.edges()
                for r in G_flow.nodes():
                    if r != root:
                        incoming = [(u, v) for u, v in arcs if v == r]
                        outgoing = [(v, w) for v, w in arcs if v == r]
                        self.addConstr(quicksum(self._f[u][v] for u, v in incoming[:])
                                       - quicksum(self._f[v][w] for v, w in outgoing[:])
                                       == self._y[v], name='F'+str(r))
            else:
                arcs = G_flow.edges()
                for r in G_flow.nodes():
                    incoming = [(u, v) for u, v in arcs if v == r]
                    outgoing = [(v, w) for v, w in arcs if v == r]
                    self.addConstr(quicksum(self._f[u][v] for u, v in incoming[:])
                                   - quicksum(self._f[v][w] for v, w in outgoing[:])
                                   >= self._y[r] - self._x[r] * (G_flow.number_of_nodes() + 1), name='F' + str(r))
        self.update()

    def add_violated_constraint(self, G, connected_components):
        """Adds violated constrains from given invalid set
        Parameters:
        G : NetworkX Graph
        connected_components : [vertex]"""

        if G.is_multigraph():
            for s in connected_components:
                elist = [e for e in G.edges(keys=True) if (e[0] in s) ^ (e[1] in s)]
                for v in s:
                    self.addConstr(self._y[v] <=  (quicksum(self._x[u] for u in s)) +
                                   (quicksum(self._z[v1][v2][k] for v1, v2, k in elist[:])), name='Conn'+str(v))
        else:
            for s in connected_components:
                elist = [e for e in G.edges() if (e[0] in s) ^ (e[1] in s)]
                for v in s:
                    self.addConstr(self._y[v] <=  (quicksum(self._x[u] for u in s)) +
                                   (quicksum(self._z[v1][v2] for v1, v2 in elist[:])), name='Conn'+str(v))
        self.update()
