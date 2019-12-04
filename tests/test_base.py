import pytest
from funkyAD.base import AD, grad, Node
from funkyAD.functions import exp 

# to do:
#  raise typeerror fo string nodes
#  raise value error for int seeds
#  update example in header to output ndarray [[1]]

# test AD class
def test_AD_string_input():
    with pytest.raises(TypeError):
        AD('hello').grad(1) 

def test_non_callable_constant_f():
    adobj = AD(2) 
    with pytest.raises(TypeError): 
        adobj.grad(2)==[[2]]

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

def test_rsub():
    assert 2-Node(1,2) == Node(1, -2)

def test_rfloordiv():
    assert 8 //  Node(3, 3) == Node(2, 0)

def test_abs():
    assert abs(Node(-1, 1)) == Node(1, -1)

def test_ne():
    assert Node(2,3) != Node(3,2)

def test_lt():
    assert Node(2,0) <= Node(2,0)

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


