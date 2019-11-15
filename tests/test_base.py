import pytest
from funkyAD.base import AD, grad, Node

def test_new_valid_function():
    def cube(x):
        return x ** 3
    
    assert grad(cube)(10) == [[300]]

def test_node_nonnumeric_values():
    with pytest.raises(TypeError):
        n = Node('Hello', 'World')
