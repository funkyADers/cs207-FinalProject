import pytest
import numpy as np
from funkyAD.base import AD, grad, Node
from funkyAD.functions import addition, multiplication, division, floordiv, power, sign, r_der, pos, neg, _abs, invert, _round, floor, ceil, trunc, exp, sin, cos, tan

# to do:
# deal with int, float, long 
# test new methods 

# test AD class
def test_noncallable_f():
    with pytest.raises(TypeError):
        AD(5*5)

def test_AD_string_input():
    with pytest.raises(TypeError):
        AD('hello').grad(1) 

def test_constant_function(): 
    adobj = AD(lambda x: 4)
    assert adobj.grad(2)==[[0]]

def test_no_args():
    adobj = AD(lambda x: x)
    with pytest.raises(TypeError):
        adobj._forward()

def test_usage_example(): 
    def f(x):
        return exp(x)
    assert AD(f).grad(0)==[[1]]

def test_grad():
    def f(x): 
        return x**2 
    assert AD(f).grad(2)==[[4]]

def test_trace():
    adobj = AD(lambda x,y: x+y)
    trace = adobj._buildtrace(1,2)
    assert len(trace)==3 

def test_trace_1dim():
    adobj = AD(lambda x: 3)
    trace = adobj._buildtrace(2)
    assert len(trace)==1

def test_reverse():
    def f(x):
        return x**2
    assert AD(f)._reverse(2)==[[4]]

def test_reverse_multidim_n():
    adobj = AD(lambda x,y: x**2+2*y)
    truth = [[4, 2]]
    assert (adobj._reverse(2,1) == truth).all()

def test_reverse_multidim_n():
    adobj = AD(lambda x,y: x**2+2*y)
    truth = [[4, 2]]
    assert (adobj._reverse(2,1) == truth).all()

def test_reverse_multidim_m(): 
    adobj = AD(lambda x,y: [x+y, x**2])
    grad = adobj._reverse(1,1)
    truth = [[1,1],[2,0]]
    assert (grad == truth).all()

def test_reverse_noinfl_in():
    adobj = AD(lambda x,y: x**2 + x)
    grad = adobj._reverse(1,1) 
    truth = [[3,0]]
    assert (grad == truth).all()

def test_set_mode():
    adobj = AD(lambda x,y: x+y)
    adobj.set_mode('reverse')
    assert adobj.mode=='reverse'

def test_set_incorrect_set_mode():
    adobj = AD(lambda x,y: x+y)
    with pytest.raises(ValueError):
        adobj.set_mode('backprop')

def test_set_incorrect_mode():
    adobj = AD(lambda x,y: x+y)
    adobj.mode = 'backprop'
    with pytest.raises(ValueError):
        adobj.grad(1,2)

def test_grad_reverse():
    def f(x):
        return x**2
    adobj = AD(f)
    adobj.set_mode('reverse')
    assert adobj.grad(2)==[[4]]

def test_grad_reverse_multidim_n():
    adobj = AD(lambda x,y: x**2+2*y)
    adobj.set_mode('reverse')
    truth = [[4, 2]]
    assert (adobj.grad(2,1) == truth).all()

def test_set_seed():
    adobj = AD(lambda x: x+5)
    adobj.set_seed(5)
    adobj.set_seed(0)
    assert adobj.seed == 0

def test_set_seed_wrong_dim():
    adobj = AD(lambda x: x+5)
    adobj.set_seed([2,1])
    with pytest.raises(ValueError):
        adobj._forward(3)

def test_set_seed_nonarray():
    adobj = AD(lambda x: x*x)
    with pytest.raises(ValueError): 
        adobj.set_seed('what')

def test_check_seed():
    adobj = AD(lambda x: 2*x+x**3)
    adobj.n = 5
    adobj.set_seed([2,1])
    with pytest.raises(ValueError): 
        adobj._check_seed(adobj.n)

def test_self_n(): 
    adobj = AD(lambda x,y: x+y)
    adobj._forward(1,2)
    assert adobj.n == 2

def test_multidim_n(): 
    adobj = AD(lambda x,y: x+y)
    grad = adobj.grad(1,1)
    truth = [[1, 1]]
    assert (grad == truth).all()

def test_multidim_m(): 
    adobj = AD(lambda x,y: [x+y, x**2])
    grad = adobj.grad(1,1)
    truth = [[1,1],[2,0]]
    assert (grad == truth).all()

def test_self_m(): 
    adobj = AD(lambda x,y: x**2 + y+y)
    adobj._forward(0,0)
    assert adobj.m == 1

def test_sum():
    def myfunc(a):
        return a.sum()
    adobj = AD(myfunc)
    inp = np.array([n for n in range(5)])
    truth = [[1, 1, 1, 1, 1]]
    assert (adobj._reverse(inp) == truth).all()

# test Node class - overload tests in test-functions.py 
def test_node_nonnumeric_values():
    with pytest.raises(TypeError):
        Node('Hello', 'World')

def test_create_node():
    node = Node(1,2)
    assert node == Node(1,2)

def test_creat_node_deriv():
    node = Node(1, [2, 3])
    assert node.d == [2, 3]

def test_default_0_deriv(): 
    assert Node(1) == Node(1,0)

def test_radd():
    assert 2+Node(1,2) == Node(3,2)

def test_rmult():
    assert 2*Node(2,2) == Node(4,4)

def test_rpow():
    with pytest.raises(NotImplementedError):
        2**Node(2,3)

def test_sub():
    assert Node(1,2)-2 == Node(-1, 2) 

def test_rsub():
    assert 2-Node(1,2) == Node(1, -2)

def test_pos():
    print(+Node(-1,-2))
    assert +(Node(-1,-2)) == Node(-1,-2)

def test_neg():
    assert -(Node(1,2)) == Node(-1, -2)

def test_rfloordiv():
    assert 8 //  Node(3, 3) == Node(2, 0)

def test_abs():
    assert abs(Node(-1, 1)) == Node(1, -1)

# invert is not differentiable 
def test_invert():
    with pytest.raises(ValueError):
        ~(Node(2, 0)) 

def test_round():
    assert round(Node(2.2, 3.2),0) == Node(2, 0)
    assert round(Node(2.2, 3.2), Node(0,0)) == Node(2, 0)

def test_floor(): 
    assert floor(Node(3.3, 3)) == Node(3,0)

def test_ceil(): 
    assert ceil(Node(3.3, 3)) == Node(4,0)

def test_trunc():
    assert trunc(Node(3.33, 3)) == Node (3,0)

# catch complex input (must be a float)
def test_complex_input():
    with pytest.raises(TypeError):
        Node(complex(3), 2)

# catch vonersion to complex
def test_complex():
    with pytest.raises(NotImplementedError):
        complex(Node(1,3))

def test_ne():
    assert Node(2,3) != Node(3,2)

def test_lt():
    assert Node(2,0) <= Node(3,0)

def test_gt():
    assert Node(3,0) >= Node(1,0)
 
def test_le():
    assert Node(2,0) < Node(3,-1)

def test_ge():
    assert Node(2,0) > Node(1,4)

def test_str():
    n = Node(2,1)
    assert str(n) == 'Node object with value 2 and derivative 1 and back-gradient 0'

def test_repr():
    n = Node(2,-1)
    assert repr(n) == 'Node(2, -1)'

# test grad function (syntatic sugar for AD) 
def test_grad_new_function():
    def cube(x):
        return x ** 3
    assert grad(cube)(10) == [[300]]


