from base import Node
import numpy as np

class BaseFunction():
    '''Defines a function that can be used on Node objects and propagate the partial derivatives

    BaseFunction(function, derivative)

    - function: function with n Node arguments and m Node outputs. Can use any operation between integers,
        including computations perfomed by secondary libraries (e.g. numpy).
    - derivative: function that (symbolically or numerically) evaluates the derivative of the
        given function. Takes Node objects as input/output. Again, can use secondary library functions.

    BaseFunction(f, d)(Node(5, 4), Node(2, 6)):
        equals: Node(f(Node(5, 4), Node(2, 6)), d(Node(5, 4), Node(2, 6)))

    Returns a Node object by applying the specified function and the specified derivative.
    '''

    def __init__(self, function, derivative):
        self.f = function
        self.d = derivative

    def __call__(self, *args):
        # Replace constants with node objects with no derivative
        new_args = [a if isinstance(a, Node) else Node(a) for a in args]

        new = Node(self.f(*new_args), self.d(*new_args))

        #new.prev = [*args]
        #for x in args:
        #    x.next.append(new)

        return new

add = BaseFunction(lambda x, y: x.v + y.v, lambda x, y: x.d + y.d)
addition = add
mul = BaseFunction(lambda x, y: x.v * y.v, lambda x, y: x.d * y.v + x.v * y.d)
multiplication = mul
div = BaseFunction(lambda x, y: x.v / y.v, (y.v * x.d - x.v * y.d) / (y.v ** 2))
division = div
_pow = BaseFunction(lambda x, n: x.v ** n.v, lambda x, n: n.v * (x.v ** (n.v - 1)) * x.d)
power = _pow
pos = BaseFunction(lambda x: +x.v, lambda x: +x.d)
neg = BaseFunction(lambda x: -x.v, lambda x: -x.d)

def invalid_op(name):
    raise ValueError("Function " + name + " is not differentiable")

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        invalid_op('abs')

_abs = BaseFunction(lambda x: abs(x.v), lambda x: x.d * sign(x.v))
invert = BaseFunction(lambda x: x.v.__invert__(), lambda x: invalid_op("__invert__"))

from math import floor, ceil, trunc
def r_der(x):
    # Derivative of rounding functions
    if ceil(x) == x and floor(x) == x:
        invalid_op("rounding")
    return 0
_round = BaseFunction(lambda x, n: round(x.v, n), r_der(x.v))
floor = BaseFunction(lambda x: floor(x.v), r_der(x.v))
ceil = BaseFunction(lambda x: ceil(x.v), r_der(x.v))
trunc = BaseFunction(lambda x: trunc(x.v), r_der(x.v))
floordiv = BaseFunction(lambda x, y: x.v // y.v, lambda x, y: r_der(x.v / y.v))

exp = BaseFunction(lambda x: np.exp(x.v), lambda x: x.d * np.exp(x.v))
sin = BaseFunction(lambda x: np.sin(x.v), lambda x: np.cos(x.v))
cos = BaseFunction(lambda x: np.cos(x.v), lambda x: -np.sin(x.v))
tan = BaseFunction(lambda x: np.tan(x,v), lambda x: 1 / (np.cos(x.v) ** 2))