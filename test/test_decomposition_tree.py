from decomposition_tree import DecompositionTree


def test_init_tree():
    D = DecompositionTree()
    
    assert D
    assert isinstance(D, DecompositionTree)
    assert not D.composition
    assert not D.left
    assert not D.right
    
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)
    
    assert D1
    assert isinstance(D1, DecompositionTree)
    assert D1.composition == 's'
    assert D1.left == D3
    assert D1.right == D4
    
    assert D2
    assert isinstance(D2, DecompositionTree)
    assert D2.composition == 'p'
    assert D2.left == D5
    assert D2.right == D6


def test_is_leaf():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)
    
    D0 = DecompositionTree('s', left=D1, right=D2)
    
    assert D3.is_leaf()
    assert D4.is_leaf()
    assert D5.is_leaf()
    assert D6.is_leaf()
    
    assert not D1.is_leaf()
    assert not D2.is_leaf()
    
    assert D0.left.left.is_leaf()
    assert D0.left.right.is_leaf()
    assert D0.right.left.is_leaf()
    assert D0.right.right.is_leaf()
    
    assert not D0.is_leaf()
    assert not D0.left.is_leaf()
    assert not D0.right.is_leaf()


def test_get_level_of_node():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    assert D0.get_level_of_node(D0) == 1
    assert D0.get_level_of_node(D1) == 2
    assert D0.get_level_of_node(D2) == 2
    assert D0.get_level_of_node(D3) == 3
    assert D0.get_level_of_node(D4) == 3
    assert D0.get_level_of_node(D5) == 3
    assert D0.get_level_of_node(D5) == 3



def test_get_depth():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    assert D0.depth() == 3
    assert D1.depth() == 2
    assert D2.depth() == 2
    assert D3.depth() == 1
    assert D4.depth() == 1
    assert D5.depth() == 1
    assert D6.depth() == 1


def test_get_leaves():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    leaves0 = D0.get_leaves()
    leaves1 = D1.get_leaves()
    leaves2 = D2.get_leaves()
    leaves3 = D3.get_leaves()
    leaves4 = D4.get_leaves()
    leaves5 = D5.get_leaves()
    leaves6 = D6.get_leaves()

    assert leaves0 == [D3, D4, D5, D6]
    assert leaves1 == [D3, D4]
    assert leaves2 == [D5, D6]
    assert leaves3 == [D3]
    assert leaves4 == [D4]
    assert leaves5 == [D5]
    assert leaves6 == [D6]


def test_size():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    assert D0.size() == 7
    assert D1.size() == 3
    assert D2.size() == 3
    assert D3.size() == 1
    assert D4.size() == 1
    assert D5.size() == 1
    assert D6.size() == 1


def test_children():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)
    
    D0 = DecompositionTree('s', left=D1, right=D2)
    
    assert D0.children() == [D1, D2]
    assert D1.children() == [D3, D4]
    assert D2.children() == [D5, D6]
    assert D3.children() == []
    assert D4.children() == []
    assert D5.children() == []
    assert D6.children() == []


def test_contains():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)
    
    D0 = DecompositionTree('s', left=D1, right=D2)
    
    assert D0.contains(D0)
    assert D0.contains(D1)
    assert D0.contains(D2)
    assert D0.contains(D3)
    assert D0.contains(D4)
    assert D0.contains(D5)
    assert D0.contains(D6)
    
    assert D1.contains(D1)
    assert D1.contains(D3)
    assert D1.contains(D4)
    assert not D1.contains(D0)
    assert not D1.contains(D2)
    assert not D1.contains(D5)
    assert not D1.contains(D6)
    
    assert D2.contains(D2)
    assert D2.contains(D5)
    assert D2.contains(D6)
    assert not D2.contains(D0)
    assert not D2.contains(D1)
    assert not D2.contains(D3)
    assert not D2.contains(D4)


def test_get_path_in_tree_as_string():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)
    
    D0 = DecompositionTree('s', left=D1, right=D2)
     
    assert D0.get_path_in_tree_as_string(D1) == 'l'
    assert D0.get_path_in_tree_as_string(D2) == 'r'
    assert D0.get_path_in_tree_as_string(D3) == 'll'
    assert D0.get_path_in_tree_as_string(D4) == 'lr'
    assert D0.get_path_in_tree_as_string(D5) == 'rl'
    assert D0.get_path_in_tree_as_string(D6) == 'rr'


def test_get_right():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    assert D0.get_right() == D2
    assert D1.get_right() == D4
    assert D2.get_right() == D6
    assert not D3.get_right()
    assert not D4.get_right()
    assert not D5.get_right()
    assert not D6.get_right()


def test_get_left():
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()

    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', left=D5, right=D6)

    D0 = DecompositionTree('s', left=D1, right=D2)

    assert D0.get_left() == D1
    assert D1.get_left() == D3
    assert D2.get_left() == D5
    assert not D3.get_left()
    assert not D4.get_left()
    assert not D5.get_left()
    assert not D6.get_left()
