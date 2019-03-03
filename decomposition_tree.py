class DecompositionTree:

    def __init__(self, composition=None, s=None, t=None, join=None, graph=None, parent=None, left=None, right=None):
        self.composition = composition
        self.s = s
        self.t = t
        self.join = join
        self.graph = graph
        self.parent = parent
        self.left = left
        self.right = right

    def set_parent(self, parent):
        self.parent = parent

    def set_graph(self, graph):
        self.graph = graph

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
        if self.children:
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

        l_depth = 0
        r_depth = 0

        if self.left:
            l_depth = self.left.depth()
            if not self.right:
                return l_depth + 1
        if self.right:
            r_depth = self.right.depth()
            if not self.left:
                return r_depth + 1
        if l_depth > r_depth:
            return l_depth + 1
        else:
            return r_depth + 1
  
    def get_level_of_node(self, node):

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

    def level_list(self, level):
        """Returns nodes at a given level"""
        if level == 1:
            return [self]
        elif level > 1:
            l = []
            if self.left:
                l.extend(self.left.level_list(level - 1))
            if self.right:
                l.extend(self.right.level_list(level - 1))
            return l

    def size(self):
        n = 1
        if self.left is not None:
            n += self.left.size()
        if self.right is not None:
            n += self.right.size()
        return n
