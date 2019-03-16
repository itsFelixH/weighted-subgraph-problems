class DecompositionTree:

    def __init__(self, composition=None, s=None, t=None, join=None, graph=None, parent=None, left=None, right=None):
        """Creates an new DecompositionTree.
        Parameters:
        composition: string ('P' or 'S', specifies composition of current tree)
        s: node (start node of the sp-graph at this tree)
        t: node (end node of the sp-graph at this tree)
        join: node (join node in a series composition)
        graph: NetworkX graph (start node of the sp-graph at this tree)
        parent: DecompositionTree (parent tree)
        left: DecompositionTree (left child)
        right: DecompositionTree (right child)

        Returns:
        DecompositionTree"""

        self.composition = composition
        self.s = s
        self.t = t
        self.join = join
        self.graph = graph
        self.parent = parent
        self.left = left
        self.right = right

    def set_parent(self, parent):
        """Sets the parent of the tree.
        Parameters:
        parent: DecompositionTree"""

        self.parent = parent

    def set_graph(self, graph):
        """Sets the sp-graph corresponding to the tree.
        Parameters:
        graph: NetworkX graph"""

        self.graph = graph

    def is_leaf(self):
        """Checks if the tree is a leaf.

        Returns:
        Boolean"""

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
        """Checks if node is in the tree.
        Parameters:
        node: DecompositionTree (node to check)

        Returns:
        Boolean"""

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
        """Returns the list of leaves in the tree.

        Returns:
        leaves: [DecompostionTree]"""

        leaves = []

        def _get_leaf_nodes(node):
            """Helper function for get_leaves.
            Recursively checks if given node is a leaf and adds it to the list of leaves.
            Parameters:
            node: DecompositionTree (node to get leaves from)"""

            if node is not None:
                if len(node.children()) == 0:
                    leaves.append(node)
                for n in node.children():
                    _get_leaf_nodes(n)

        _get_leaf_nodes(self)
        return leaves
    
    def depth(self):
        """Returns depth of the tree.

        Returns:
        depth: int"""

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
        """Returns level of given node in the tree.
        Parameters:
        node: DecompositionTree (node to get level from)

        Returns:
        int (level of the node in the tree)"""

        def _get_level_util(tree, node, level):
            """Helper function for get_level_of_node.
            Recursively checks if given level contains the node.
            Parameters:
            tree: DecompositionTree
            node: DecompositionTree (node to get level from)
            level: int

            Returns:
            down_level : int (level of the node in the tree)"""

            if not tree:
                return 0
          
            if tree == node:
                return level
          
            down_level = _get_level_util(tree.left, node, level + 1)
            if down_level != 0:
                return down_level
          
            down_level = _get_level_util(tree.right, node, level + 1)
            return down_level

        return _get_level_util(self, node, 1)
    
    def get_left(self):
        """Returns left child of the tree node.

        Returns:
        node (DecompositionTrees)"""

        if self.left is not None:
            return self.left
    
    def get_right(self):
        """Returns right child of the tree node.

        Returns:
        node (DecompositionTrees)"""

        if self.right is not None:
            return self.right
    
    def children(self):
        """Returns children of the tree node.

        Returns:
        child_list : list of nodes (DecompositionTrees)"""

        child_list = []
        if self.left is not None:
            child_list.append(self.left)
        if self.right is not None:
            child_list.append(self.right)
        return child_list

    def level_list(self, level):
        """Returns nodes at a given level of the tree.
        Parameters:
        level: int (level of desired nodes)

        Returns:
        l : list of nodes (DecompositionTrees)"""

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
