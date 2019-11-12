import numpy as np

class AD():
    '''Wraps a function to access Automatic Differentiation methods.

    AD(lambda x, y: x ** 2 - y)
    -   creates an AD object that wraps a speficied function. Function can be given in
        extended form (using def f(...)) or short (using lambdas, as above). 
        Function cantake an arbitrary number of inputs by passing several arguments, 
        or an iterable (list or np.array) containing the argument. The number of inputs 
        can vary each time the function is called.
        Function can return an arbitrary number of outputs as an iterable. The number of 
        outputs can vary from call to call depending on input.
        Function can use any of the standard arithmetic operators, or BaseFunctions. Most
        of these are already defined for you in the module functions.py.

    Sample usage:
    >>> from base import AD
    >>> from functions import exp
    >>> def f(x):
            return exp(x)
    >>> print(AD(f).grad(0))
    1
    '''

    def __init__(self, f):
        self.f = f
        self.seed = None
        self.n = None
        self.m = None

    def grad(self, *args):
        '''Returns the gradient of the function evaluated on the arguments given'''
        out_nodes = self._evaluate(*args)
        return np.array([node.d for node in unpack(out_nodes)])

    def set_seed(self, seed):
        '''Sets a matrix of seed vectors for the forward mode pass.
        If n inputs and m outputs are given, seed argument must have (n, m) shape.
        '''
        self.seed = seed

    def _check_seed(self, l):
        '''Checks if provided seed has appropriate dimension'''
        if len(seed) != l:
            raise ValueError("Seed dimension does not match input dimension")

    def _evaluate(self, *args):
        '''Main algorithm for the forward pass. Calls helpers as appropriate.'''

        # Compute number of inputs
        self.n = count_recursive(args)

        # Try evaluate the function
        output = self.f(*args)
        # Compute numer of outputs
        self.m = count_recursive(output)

        # Set seed if not supplied by user
        if self.seed is not None:
            default_seed = 0
            self._check_seed(self.n)
        else:
            default_seed = 1
            self.set_seed(np.eye(self.n)) # identity matrix by default

        # Make all arguments Node objects
        new_args = nodify(args, self.seed)

        if default_seed:
            # If we assigned a default seed, remove it
            self.seed = None

        return self.f(*new_args)


class Node():
    '''Represents a Node in the evaluation graph. Holds its value and derivative. 

    During evaluation, Nodes are dynamically created and destroyed as necessary. Trace is
        stored by having each Node remember its parents and siblings.

    Node(4, [1, 0, 0]):
        creates a node object with value 4 and derivative [1, 0, 0]
    '''

    def __init__(self, v, d=0):
        self.v = v
        self.d = d # If derivative is None it's assumed to be 0 (for a constant node)

        #self.prev = []
        #self.next = []

    def __add__(self, other):
        return addition(self, other)
    def __radd__(self, other):
        return addition(self, other)
    def __sub__(self, other):
        return addition(self, -other)
    def __rsub__(self, other):
        return addition(-self, other)
    def __mul__(self, other):
        return multiplication(self, other)
    def __rmul__(self, other):
        return multiplication(self, other)
    def __pow__(self, other):
        return power(self, other)
    def __rpow__(self, other):
        ## TODO
        raise NotImplemented
    def __floordiv__(self, other):
        return floordiv(self, other)
    def __rfloordiv__(self, other):
        return floordiv(other, self)
    def __div__(self, other):
        return division(self, other)
    def __rdiv__(self, other):
        return division(other, self)

    # TODO
    #def __iadd__(self, other): etc

    def __pos__(self):
        return pos(self)
    def __neg__(self):
        return neg(self)
    def __abs__(self):
        return _abs(self)
    def __invert__(self):
        return invert(self)
    def __round__(self, n):
        return _round(self, n)
    def __floor__(self):
        return floor(self)
    def __ceil__(self):
        return ceil(self)
    def __trunc__(self):
        return trunc(self)


    def __eq__(self, other):
        return self.v.__eq__(other.v)
    def __ne__(self, other):
        return self.v.__ne__(other.v)
    def __lt__(self, other):
        return self.v.__lt__(other.v)
    def __gt__(self, other):
        return self.v.__gt__(other.v)
    def __le__(self, other):
        return self.v.__le__(other.v)
    def __ge__(self, other):
        return self.v.__ge__(other.v)

    def __int__(self):
        return self.__floor__(self)
    def __long__(self):
        return self.__floor__(self)   
    def __float__(self):
        return self
    def __complex__(self):
        raise NotImplemented("Complex number not supported")
        
    def __str__(self):
        return "Node object with value " + str(self.v) + " and derivative " + str(self.d)

    def __repr__(self):
        return "Node(" + str(self.v) + ", " + str(self.d) + ")"


    def __pos__(self):
        return 

    def __eq__(self, other):
        return self.v.__eq__(other.v)
    def __ne__(self, other):
        return self.v.__ne__(other.v)
    def __lt__(self, other):
        return self.v.__lt__(other.v)
    def __gt__(self, other):
        return self.v.__gt__(other.v)
    def __le__(self, other):
        return self.v.__le__(other.v)
    def __ge__(self, other):
        return self.v.__ge__(other.v)



        
    def __str__(self):
        return "Node object with value " + str(self.v) + " and derivative " + str(self.d)

    def __repr__(self):
        return "Node(" + str(self.v) + ", " + str(self.d) + ")"


def grad(f):
    '''Syntactic sugar for AD(f).grad.

    Sample usage:
    >>> from base import grad
    >>> def f(x, y):
            return x + y
    >>> print(grad(f)(3, 5))
    [1, 1]
    '''
    return AD(f).grad

# Import statement has to be at the bottom for some reason
from funkyAD.functions import addition, multiplication, division, power, pos, neg, _abs, invert, \
                        floordiv, _round, floor, ceil, trunc
from funkyAD.helpers import count_recursive, nodify, unpack

if __name__ == "__main__":
    print(Node(4, 5))
    print(AD(lambda x: x ** 2))
