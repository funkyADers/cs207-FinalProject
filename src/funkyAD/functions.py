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
        # Deferred import to work around circular dependencies
        from .base import Node
        # Replace constants with node objects with no derivative
        new_args = [a if isinstance(a, Node) else Node(a) for a in args]

        new = Node(self.f(*new_args), self.d(*new_args))

        return new

addition = BaseFunction(lambda x, y: x.v + y.v, lambda x, y: x.d + y.d)
multiplication = BaseFunction(lambda x, y: x.v * y.v, lambda x, y: x.d * y.v + x.v * y.d)
division = BaseFunction(lambda x, y: x.v / y.v, lambda x, y: (y.v * x.d - x.v * y.d) / (y.v ** 2))
power = BaseFunction(lambda x, n: x.v ** n.v, lambda x, n: n.v * (x.v ** (n.v - 1)) * x.d)
pos = BaseFunction(lambda x: +x.v, lambda x: +x.d)
neg = BaseFunction(lambda x: -x.v, lambda x: -x.d)

# Function to raise error if function is non-differentiable
def invalid_op(name):
    raise ValueError("Function '" + name + "' is not differentiable")

# Get sign of a value
def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        invalid_op('abs')

_abs = BaseFunction(lambda x: abs(x.v), lambda x: x.d * sign(x.v))
invert = BaseFunction(lambda x: x.v.__invert__(), lambda x: invalid_op("__invert__"))

# The derivative of the floor of x is 0 when x is non-integer and not defined when it is
def r_der(x):
    # Derivative of rounding functions
    if np.ceil(x) == x and np.floor(x) == x:
        # Derivative of floor of integer is not mathematically defined
        invalid_op('rounding')
    return 0

def round1(x, n=0):
    from .base import Node
    if isinstance(n, Node):
        n = n.v
    return np.round(x.v * (10 ** n)) / (10. ** n)
def round2(x, n=0):
    from .base import Node
    if isinstance(n, Node):
        n = n.v
    return r_der(x.v * (10 ** n))
_round = BaseFunction(round1, round2)
floor = BaseFunction(lambda x: np.floor(x.v), lambda x: r_der(x.v))
ceil = BaseFunction(lambda x: np.ceil(x.v), lambda x: r_der(x.v))
trunc = BaseFunction(lambda x: np.trunc(x.v), lambda x: r_der(x.v))

floordiv = BaseFunction(lambda x, y: x.v // y.v, lambda x, y: r_der(x.v / y.v))

exp = BaseFunction(lambda x: np.exp(x.v), lambda x: x.d * np.exp(x.v))
sin = BaseFunction(lambda x: np.sin(x.v), lambda x: x.d * np.cos(x.v))
cos = BaseFunction(lambda x: np.cos(x.v), lambda x: -x.d * np.sin(x.v))
tan = BaseFunction(lambda x: np.tan(x.v), lambda x: x.d / (np.cos(x.v) ** 2))

#if __name__ == '__main__':
#    print(exp(Node(1, 2)))