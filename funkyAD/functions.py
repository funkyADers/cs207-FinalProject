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

addition = BaseFunction(lambda x, y: x.v + y.v, lambda x, y: x.d + y.d)
multiplication = BaseFunction(lambda x, y: x.v * y.v, lambda x, y: x.d * y.v + x.v * y.d)
power = BaseFunction(lambda x, n: x.v ** n.v, lambda x, n: n.v * (x.v ** (n.v - 1)) * x.d)
exp = BaseFunction(lambda x: np.exp(x.v), lambda x: x.d * np.exp(x.v))

