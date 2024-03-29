import pytest
import numpy as np
from funkyAD.base import Node
from funkyAD.functions import BaseFunction, invalid_op, addition, multiplication, division, floordiv, power, sqrt, sign, r_der, pos, neg, _abs, invert, _round, floor, ceil, trunc, base_check, exp, log, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, one_check, logistic, sigmoid

# BaseFunction
def test_define_basefunction():
    # Try both lambda and custom function
    def my_derivative(x, y):
        return x * y
    BaseFunction(lambda x, y: x.v + y.v, my_derivative)

def test_use_new_basefunction():
    new_func = BaseFunction(lambda x, y: x.v + y.v, lambda x, y: x.d * y.d)
    assert new_func(Node(1, 2), Node(3, 4)) == Node(4, 8)

def test_invalid_op():
    with pytest.raises(ValueError):
        invalid_op('test_call')

# Only need one overload test each; if that works, all other variations will too

# If one function on invalid Node raises correct error, all will
def test_addition_invalid_node():
    with pytest.raises(TypeError):
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

# Subtraction implemented indirectly by combining addition and negation
def test_subtraction():
    # Two nodes via function call
    assert addition(Node(3, 4), -Node(1, 2)) == Node(2, 2)
    # Two nodes via overload
    assert Node(3, 4) - Node(1, 2) == Node(2, 2)
    # Two constants via function call
    assert addition(2, -1) == Node(1, 0)
    # Node and constant
    assert addition(Node(3, 2), -2) == Node(1, 2)

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
    ## 3*(2**2)*2
    assert power(Node(2, 2), Node(3, 3)) == Node(8, 40.63553233343869)
    # Node and constant
    assert power(Node(2, 2), 3) == Node(8, 24)
    # Two nodes via overload
    assert Node(2, 2) ** Node(3, 3) == Node(8, 40.63553233343869)
    # Two constants via function call
    assert power(4, 2) == Node(16, 0)
    # Node and constant
    assert Node(3, 3) ** 2 == Node(9, 18)
    # Negative power (not invert, but chain rule)
    assert Node(2, 3) ** -2 == Node(0.25, -0.75)

def test_sqrt():
    assert sqrt(Node(2, 3)) == Node(2**0.5, 3/(2*2**0.5))
    assert sqrt(2) == Node(2**0.5, 0)

def test_pos():
    assert pos(Node(1, 2)) == Node(1, 2)
    assert pos(Node(-1, -2)) == Node(-1, -2)
    assert pos(2) == Node(2, 0)
    assert pos(-2) == Node(-2, 0)
    
# Test negative (negation)
def test_negation():
    assert neg(Node(1, 2)) == Node(-1, -2)
    assert neg(Node(-1, -2)) == Node(1, 2)
    assert neg(2) == Node(-2, 0)
    assert neg(-2) == Node(2, 0)

def test_sign():
    assert sign(-5) == sign(-2.3) == -1
    assert sign(5) == sign(2.3) == 1

def test_sign_abs():
    with pytest.raises(ValueError):
        sign(0)

def test_r_der_nonint():
    assert r_der(5.5) == 0
    assert r_der(-1.3) == 0

def test_r_der_int():
    with pytest.raises(ValueError):
        r_der(3)
    
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
    # Not specifying num digits
    assert _round(Node(2.56, 5.55)) == Node(3, 0)
    assert _round(1.5) == Node(2, 0)
    # Specifying num digits
    assert _round(Node(2.56, 5.55), 1) == Node(2.6, 0)
    assert _round(1.53, 1) == Node(1.5, 0)
    assert _round(1.539, 2) == Node(1.54, 0)

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

def test_base_check():
    check = base_check(np.log)
    with pytest.raises(ValueError):
        check(Node(1, 2), -2)

# Natural exponential: using default base = np.e
def test_exp():
    # Nodes
    assert exp(Node(2, 0)) == Node(np.exp(2), 0)
    assert exp(Node(2, 2)) == Node(np.exp(2), 2*np.exp(2))
    assert exp(Node(-2, -2)) == Node(np.exp(-2), -2*np.exp(-2))
    # Constants
    assert exp(1) == Node(np.exp(1), 0)
    assert exp(-1) == Node(np.exp(-1), 0)

# General exponential, base != np.e
def test_exp_general():
    # Node and constant
    assert exp(Node(2, 3), 2) == Node(2**2, 3*np.log(2)*2**2)
    # Node and base as Node
    assert exp(Node(2, 3), Node(2, 3)) == Node(2**2, 3*np.log(2)*2**2)
    
def test_exp_type():
    # Type checking still handled by Node constructor, but verify it works
    with pytest.raises(TypeError):
        exp(Node(1,2), 'base2')

def test_exp_invalid():
    with pytest.raises(ValueError):
        exp(Node(1,2), -2)

def test_log_natural():
    # Full node
    assert log(Node(np.exp(2), 4)) == Node(2, 4/np.exp(2))
    # Constant
    assert log(np.exp(2)) == Node(2, 0)

def test_log_general():
    # Node with custom base
    assert log(Node(8, 4), 2) == Node(3, 0.5/np.log(2))
    assert log(Node(9, 27), 3) == Node(2, 3/np.log(3))
    # Constant with custom base
    assert log(4, 2) == Node(2, 0)

def test_sin():
    assert sin(Node(2, 3)) == Node(np.sin(2), 3*np.cos(2))
    assert sin(2) == Node(np.sin(2), 0)

def test_arcsin():
    assert arcsin(Node(0.5, 1.5)) == Node(np.arcsin(0.5), 1.5/np.sqrt(1 - 0.5**2))
    assert arcsin(0.75) == Node(np.arcsin(0.75), 0)

def test_inv_trig_invalid():
    with pytest.raises(ValueError):
        arcsin(1.5)
    with pytest.raises(ValueError):
        arccos(1.5)
    with pytest.raises(ValueError):
        arctan(1.5)

def test_one_check():
    check = one_check(lambda x: x**2)
    with pytest.raises(ValueError):
        check(Node(1.5, 1.5))

def test_cos():
    assert cos(Node(2, 3)) == Node(np.cos(2), -3*np.sin(2))
    assert cos(2) == Node(np.cos(2), 0)

def test_arccos():
    assert arccos(Node(0.5, 2)) == Node(np.arccos(0.5), -(2 / np.sqrt(1 - 0.5**2)))
    assert arccos(0.85) == Node(np.arccos(0.85), 0)

def test_tan():
    assert tan(Node(2, 3)) == Node(np.tan(2), 3/np.cos(2)**2)
    assert tan(2) == Node(np.tan(2), 0)

def test_arcan():
    assert arctan(Node(0.5, 2)) == Node(np.arctan(0.5), 2/(1+0.5**2))
    assert arctan(0.5) == Node(np.arctan(0.5), 0)

def test_sinh():
    assert sinh(Node(2, 3)) == Node(np.sinh(2), 3*np.cosh(2))
    assert sinh(2) == Node(np.sinh(2), 0)

def test_cosh():
    assert cosh(Node(2, 3)) == Node(np.cosh(2), 3*np.sinh(2))
    assert cosh(2) == Node(np.cosh(2), 0)

def test_tanh():
    assert tanh(Node(2, 3)) == Node(np.tanh(2), 3/((np.cosh(2)**2)))
    assert tanh(2) == Node(np.tanh(2), 0)

def test_logistic():
    assert logistic(Node(2, 3), 10, 100) == Node(10/(1+100*np.exp(-2)), 3*10*100*np.exp(2)/(100+np.exp(2))**2)

def test_sigmoid():
    assert sigmoid(Node(2, 3)) == Node(1/(1+np.exp(-2)), 3*np.exp(2)/(1+np.exp(2))**2)
    assert sigmoid(2) == Node(1/(1+np.exp(-2)), 0)
