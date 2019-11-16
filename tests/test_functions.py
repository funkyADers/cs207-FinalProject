import pytest
from funkyAD.base import Node
from funkyAD.functions import addition, multiplication, division, floordiv

# No need to test for invalid Nodes, those are handled in the creation of Nodes

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
    assert multiplication(Node(1, 2), Node(3, 4)) == Node(3, 10)

# Multiplication overload
def test_node_multiply_overload():
    assert Node(1, 2) * Node(3, 4) == Node(3, 10)

# Multiplication constants
def test_multiply_constants():
    assert multiplication(1, 2) == Node(2, 0)

# Division valid Nodes
def test_node_division():
    assert division(Node(5, 3), Node(2, 4)) == Node(2.5, -3.5)

# Division overload
def test_node_division_overload():
    assert Node(5, 3) / Node(2, 4) == Node(2.5, -3.5)

# Division constants
def test_division_constants():
    assert division(4, 2) == Node(2, 0)

# Floor division valid Nodes
def test_node_floordivision():
    assert floordiv(Node(5, 3), Node(2, 4)) == Node(2.5, -3.5)

# Floor division overload
def test_node_floordivision_overload():
    assert Node(5, 3) // Node(2, 4) == Node(2.5, -3.5)

# Floor division constants
def test_floordivision_constants():
    assert floordiv(4, 2) == Node(2, 0)
