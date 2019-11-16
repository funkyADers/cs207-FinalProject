import pytest
from funkyAD.base import Node
from funkyAD.functions import addition, multiplication, division, floordiv, power

# No need to test for invalid Nodes, those are handled in the creation of Nodes
# Only need one overload test each; if that works, all other variations via overloads will too

def test_addition():
    # Two nodes via function call
    assert addition(Node(1, 2), Node(3, 4)) == Node(4, 6)
    # Two nodes via overload
    assert Node(1, 2) + Node(3, 4) == Node(4, 6)
    # Two constants via function call
    assert addition(1, 2) == Node(3, 0)
    # Node and constant
    assert addition(Node(1, 2), 2) == Node(3, 2)

def test_multiplication():
    # Two nodes via function call
    assert multiplication(Node(1, 2), Node(3, 4)) == Node(3, 10)
    # Two nodes via overload
    assert Node(1, 2) * Node(3, 4) == Node(3, 10)
    # Two constants via function call
    assert multiplication(1, 2) == Node(2, 0)
    # Node and constant
    assert multiplication(Node(2, 2), 2) == Node(4, 4)
    
def test_division():
    # Two nodes via function call
    assert division(Node(5, 3), Node(2, 4)) == Node(2.5, -3.5)
    # Two nodes via overload
    assert Node(5, 3) / Node(2, 4) == Node(2.5, -3.5)
    # Two constants via function call
    assert division(4, 2) == Node(2, 0)
    # Node and constant
    assert division(Node(4, 4), 2) == Node(4, 4) / 2 == Node(2, 2)
    assert 6 / Node(3, 3) == Node(2, -2)

def test_floordivision():
    # Two nodes via function call
    assert floordiv(Node(8, 8), Node(3, 3)) == Node(2, 0)
    # Two nodes via overload
    assert Node(8, 8) // Node(3, 3) == Node(2, 0)
    # Two constants via function call
    assert floordiv(11, 3) == Node(3, 0)
    # Node and constant
    assert floordiv(Node(7.5, 3.2), 2) == Node(7.5, 3.2) // 2 == Node(3, 0)

def test_power():
    # Two nodes via function call
    assert power(Node(2, 2), Node(3, 3)) == Node(8, 24)
    # Two nodes via overload
    assert Node(2, 2) ** Node(3, 3) == Node(8, 24)
    # Two constants via function call
    assert power(4, 2) == Node(16, 0)
    # Node and constant
    assert Node(3, 3) ** 2 == Node(9, 18)
