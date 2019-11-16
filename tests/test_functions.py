import pytest
import numpy as np
from funkyAD.base import Node
from funkyAD.functions import addition, multiplication, division, floordiv, power, sign, pos, neg, _abs, invert, _round, floor, ceil, trunc, exp, sin, cos, tan

# Only need one overload test each; if that works, all other variations will too

# If one function on invalid Node raises correct error, all will
def test_addition_invalid_node():
    with pytest.raises(ValueError):
        addition(Node('Text', 1), 2)

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
    # Negative power (not invert, but chain rule)
    assert Node(2, 3) ** -2 == Node(0.25, -0.75)

def test_pos():
    assert pos(Node(1, 2)) == Node(1, 2)
    assert pos(Node(-1, -2)) == Node(-1, -2)
    assert pos(2) == Node(2, 0)
    assert pos(-2) == Node(-2, 0)
    
def test_neg():
    assert neg(Node(1, 2)) == Node(-1, -2)
    assert neg(Node(-1, -2)) == Node(1, 2)
    assert neg(2) == Node(-2, 0)
    assert neg(-2) == Node(2, 0)

def test_sign():
    assert sign(-5) == sign(-2.3) == -1
    assert sign(5) == sign(2.3) == 1
    
def test_abs():
    assert _abs(Node(2, 3)) == Node(2, 3)
    assert _abs(Node(-2, 3)) == Node(2, -3)
    assert _abs(Node(2, -3)) == Node(2, -3)
    assert _abs(Node(-2, -3)) == Node(2, 3)

def test_invert():
    # Invert is not differentiable, verify that it raises the expected error
    with pytest.raises(ValueError):
        invert(Node(1, 2))

def test_round():
    assert _round(Node(2.56, 5.55), 1) == Node(2.6, 0)
    assert _round(Node(2.56, 5.55)) == Node(3, 0)

def test_floor():
    # Nodes
    assert floor(Node(3.3, 3)) == Node(3, 0)
    assert floor(Node(3.9, 3)) == Node(3, 0)
    assert floor(Node(-2.5, -2)) == Node(-3, 0)
    # Constants
    assert floor(3.3) == Node(3, 0)
    assert floor(-3.2) == Node(-4, 0)

def test_floor_undefined():
    with pytest.raises(ValueError):
        floor(Node(2, 2))

def test_ceil():
    # Nodes
    assert ceil(Node(4.1, 3)) == Node(5, 0)
    assert ceil(Node(-2.5, -2)) == Node(-2, 0)
    # Constants
    assert ceil(5.1) == Node(6, 0)
    assert ceil(-3.5) == Node(-3, 0)

def test_ceil_undefined():
    with pytest.raises(ValueError):
        ceil(Node(2, 2))

def test_trunc():
    # Nodes
    assert trunc(Node(2.1, 0)) == Node(2, 0)
    assert trunc(Node(2.9, 0)) == Node(2, 0)
    assert trunc(Node(-1.5, 0)) == Node(-1, 0)
    # Constants
    assert trunc(1.5) == Node(1, 0)
    assert trunc(-1.5) == Node(-1, 0)

def test_trunc_undefined():
    with pytest.raises(ValueError):
        trunc(Node(1, 1))

def test_exp():
    # Nodes


    # Constants
    assert exp(1) == Node(np.exp(1), 0)
    assert exp(-1) == Node(np.exp(-1), 0)

def test_sin():
    # Two nodes via function call
    pass
    # Two nodes via overload
    
    # Two constants via function call
    
    # Node and constant

def test_cos():
    # Two nodes via function call
    pass
    # Two nodes via overload
    
    # Two constants via function call
    
    # Node and constant

def test_tan():
    # Two nodes via function call
    pass
    # Two nodes via overload
    
    # Two constants via function call
    
    # Node and constant
