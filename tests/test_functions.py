import pytest
from funkyAD.base import Node
from funkyAD.functions import *

def test_node_add():
    assert add(Node(1, 2), Node(3, 4)) == Node(4, 6)

def test_node_addition_overload():
    assert Node(1, 2) + Node(3, 4) == Node(4, 6)

def test_add_constants():
    assert add(1, 2) == Node(3, 0)
