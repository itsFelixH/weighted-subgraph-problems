import graph_helper as gh


class DecompositionTree:

    def __init__(self, composition=None, joinNode=None, left=None, right=None):
        self.composition = composition
        self.joinNode = joinNode
        self.left = left
        self.right = right
        
    def get_graph(self, node, G):
        path = self.get_path_in_tree_as_string(node)
        currentNode = self
        while path != '':
            if path[0] == 'l':
                currentNode = self.left
                if currentNode.composition == 'S':
                    G = gh.createSubgraph(G, currentNode.joinNode[0], 'predecessors')
                else:
                    s = currentNode.joinNodes[0]
                    t = currentNode.joinNodes[1]
                    (G, H) = gh.createSubgraphsBetweenNodes(G, s, t)
            elif path[0] == 'r':
                currentNode = self.right
                if currentNode.composition == 'S':
                    G = gh.createSubgraph(G, currentNode.joinNode[0])
                else:
                    s = currentNode.joinNodes[0]
                    t = currentNode.joinNodes[1]
                    (H, G) = gh.createSubgraphsBetweenNodes(G, s, t)
            path = path[1:]
        return G

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        else:
            return False
    
    def get_path_in_tree_as_string(self, node):       
        if self.left and self.left.contains(node):
            return 'l' + self.left.get_path_in_tree_as_string(node)
        elif self.right and self.right.contains(node):
            return 'r' + self.right.get_path_in_tree_as_string(node)
        elif self == node:
            return ''
        else:
            return 'node is not in tree!'
    
    def contains(self, node):
        if self == node:
            return True
        if self.children != []:
            in_left_tree = False
            in_right_tree = False
            if self.left:
                in_left_tree = self.left.contains(node)
            if self.right:
                in_right_tree = self.right.contains(node)
            return in_left_tree or in_right_tree
        else:
            return False       

    def get_leaves(self):
        leaves = []
        def _get_leaf_nodes(node):
            if node is not None:
                if len(node.children()) == 0:
                    leaves.append(node)
                for n in node.children():
                    _get_leaf_nodes(n)
        _get_leaf_nodes(self)
        return leaves
    
    def depth(self):
        if not self.children():
            return 1

        lDepth = 0
        rDepth = 0

        if self.left:
            lDepth = self.left.depth()
            if not self.right:
                return lDepth + 1
        if self.right:
            rDepth = self.right.depth()
            if not self.left:
                return rDepth + 1
        if lDepth > rDepth:
            return lDepth + 1
        else:
            return rDepth + 1
  
    def get_level(self, node):
        def _get_level_util(tree, node, level): 
            if not tree:
                return 0
          
            if tree == node:
                return level
          
            downlevel = _get_level_util(tree.left, node, level + 1)  
            if downlevel != 0:
                return downlevel
          
            downlevel = _get_level_util(tree.right, node, level + 1)  
            return downlevel
        return _get_level_util(self, node, 1)
    
    def get_left(self):
        if self.left is not None:
            return self.left
    
    def get_right(self):
        if self.right is not None:
            return self.right
    
    def children(self):
        childlist = []
        if self.left is not None:
            childlist.append(self.left)
        if self.right is not None:
            childlist.append(self.right)
        return childlist
        
    def size(self):
        n = 1
        if self.left is not None:
            n += self.left.size()
        if self.right is not None:
            n += self.right.size()
        return n
