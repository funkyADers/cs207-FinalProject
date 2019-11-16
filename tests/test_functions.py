import pytest
from funkyAD.base import Node
from funkyAD.functions import add, addition, mul, multiplication

# Addition valid Nodes
def test_node_add():
    assert addition(Node(1, 2), Node(3, 4)) == Node(4, 6)

# Addition overload
def test_node_add_overload():
    assert Node(1, 2) + Node(3, 4) == Node(4, 6)

# Addition constants
def test_add_constants():
    assert addition(1, 2) == Node(3, 0)

# Multiplication valid Nodes
def test_node_multiply():
    assert mul(Node(1, 2), Node(3, 4)) == Node(3, 10)

# Multiplication overload
def test_node_multiply_overload():
    assert Node(1, 2) * Node(3, 4) == Node(3, 10)

# Multiplication constants
def test_multiply_constants():
    assert mul(1, 2) == Node(2, 0)