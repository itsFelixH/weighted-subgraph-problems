from decomposition_tree import DecompositionTree


def test_init_tree():
    D = DecompositionTree()
    
    assert D
    assert isinstance(D, DecompositionTree)
    assert D.composition == None
    assert D.joinNode == None
    assert D.left == None
    assert D.right == None
    
    D3 = DecompositionTree()
    D4 = DecompositionTree()
    D5 = DecompositionTree()
    D6 = DecompositionTree()
    D1 = DecompositionTree('s', left=D3, right=D4)
    D2 = DecompositionTree('p', joinNode=4, left=D5, right=D6)
    
    assert D1
    assert isinstance(D1, DecompositionTree)
    assert D1.composition == 's'
    assert not D1.joinNode
    assert D1.left == D3
    assert D1.right == D4
    
    assert D2
    assert isinstance(D2, DecompositionTree)
    assert D2.composition == 'p'
    assert D2.joinNode == 4
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
