import numpy as np
from .functions import addition, multiplication, division, power, pos, neg, _abs, invert, floordiv, _round, floor, ceil, trunc
from .helpers import count_recursive, nodify, unpack, recursive_append

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
    [[1]]
    '''

    def __init__(self, f):
        if not callable(f):
            raise TypeError('The input function to AD must be callable')
        self.f = f
        self.seed = None
        self.n = None
        self.m = None
        self.trace = None
        self.mode = 'forward'

    def set_mode(self, mode):
        if mode not in ('forward','reverse'):
            raise ValueError('Invalid mode = Only "forward" and "reverse" mode are supported')
        else:
            self.mode = mode

    def grad(self, *args):
        '''Returns the gradient of the function evaluated on the arguments given'''
        if self.mode not in ('forward','reverse'):
            raise ValueError('Invalid mode = Only "forward" and "reverse" mode are supported')
        if self.mode=='forward':
            out_nodes = self._forward(*args)
            return np.array([node.d for node in unpack(out_nodes)])
        elif self.mode == 'reverse':
            out_nodes = self._reverse(*args)
            return out_nodes

    def set_seed(self, seed):
        '''Sets a matrix of seed vectors for the forward mode pass.
        If n inputs and m outputs are given, seed argument must have (n, m) shape.
        '''
        try:
            self.seed = np.asarray(seed,dtype=np.float32)
        except: 
            raise ValueError("Seed input must be an array with numeric elements")

    def _check_seed(self, l):
        '''Checks if provided seed has appropriate dimension'''
        if len(self.seed) != l:
            raise ValueError("Seed dimension does not match input dimension")

    def _forward(self, *args):
        '''Main algorithm for the forward pass. Calls helpers as appropriate.'''

        # Compute number of inputs
        self.n = count_recursive(args)

        # Try to evaluate the function
        try:
            output = self.f(*args)
        except:
            raise TypeError('function and *args are not callable') 
            
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
        self.input_nodes = new_args

        if default_seed:
            # If we assigned a default seed, remove it
            self.seed = None

        # Replace constants in the output with Node objects
        out = self.f(*new_args)
        if hasattr(type(out), '__len__'):
            self.output_nodes = np.array([a if isinstance(a, Node) else Node(a) for a in out])
        else:
            self.output_nodes = np.array([out]) if isinstance(out, Node) else np.array([Node(out)])
        return self.output_nodes

    def _buildtrace(self, *args):
        
        # Store previous seed in a temp value, set seed to 0
        temp, self.seed = self.seed, [0 for _ in range(count_recursive(args))]
        out = self._forward(*args)
        # Put it back
        self.seed = temp

        trace = []

        # Nodify 
        # self._forward will always return an np.array 
        for a in out:
            recursive_append(a, trace)

        self.trace = trace
        return trace

    def _reverse(self, *args):

        trace = self._buildtrace(*args)

        # Set df/dx_n of the output to 1
        self.back_seed = np.eye(len(self.output_nodes))
        for i, n in enumerate(self.output_nodes):
            n.back_g = self.back_seed[i]

        for n in trace:
            if n.parents:
                for p in n.parents:
                    # Set derivative of that node to 1
                    p.d = 1.0
                    # Increase the gradient of the parent
                    p.back_g += n.back_g * n.f.d(*n.parents)
                    # Put it back so other nodes' computations are not affected
                    p.d = 0.0

        # Unpack input Node objects in case they are contained in an array or similar
        new_input = []
        for var in self.input_nodes:
            if hasattr(type(var), '__len__'):
                new_input += [x for x in var]
            else:
                new_input.append(var)

        for n in new_input:
            # If input node does not influence the output, its gradient has to be
            #  manually set to have the correct size
            try:
                if n.back_g == 0:
                    n.back_g = np.zeros(len(self.output_nodes))
            except:
                pass

        # Return the transpose for (obvious?) mathematical reasons
        return np.array([x.back_g for x in new_input]).T

class Node():
    '''Represents a Node in the evaluation graph. Holds its value and derivative. 

    During evaluation, Nodes are dynamically created and destroyed as necessary. Trace is
        stored by having each Node remember its parents and siblings.

    Node(4, [1, 0, 0]):
        creates a node object with value 4 and derivative [1, 0, 0]
    '''

    def __init__(self, v, d=0):
        self.v = v
        self.d = d # if derivative is none its assumed to be 0 (for constant node)

        # Verify that numeric
        try:
            np.asarray(self.v, dtype=np.float64)
            np.asarray(self.d, dtype=np.float64)
        except:
            raise TypeError('Value and derivative must be numeric.')

        self.parents = None
        self.f = None
        self.back_g = 0 #Backprop gradient


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
        raise NotImplementedError()
    def __floordiv__(self, other):
        return floordiv(self, other)
    def __rfloordiv__(self, other):
        return floordiv(other, self)
    def __truediv__(self, other):
        return division(self, other)
    def __rtruediv__(self, other):
        return division(other, self)
    def __pos__(self):
        return pos(self)
    def __neg__(self):
        return neg(self)
    def __abs__(self):
        return _abs(self)
    def __invert__(self):
        return invert(self)
    def __round__(self, n):
        if isinstance(n, Node):
            return _round(self, n.v)
        return _round(self, n)
    def __floor__(self):
        return floor(self)
    def __ceil__(self):
        return ceil(self)
    def __trunc__(self):
        return trunc(self)

    def __eq__(self, other):
        return self.v.__eq__(other.v) and self.d.__eq__(other.d)
    def __ne__(self, other):
        return self.v.__ne__(other.v) or self.d.__ne__(other.d)

    def __lt__(self, other):
        return self.v.__lt__(other.v)
    def __gt__(self, other):
        return self.v.__gt__(other.v)
    def __le__(self, other):
        return self.v.__le__(other.v)
    def __ge__(self, other):
        return self.v.__ge__(other.v)
    def __complex__(self):
        raise NotImplementedError("Complex numbers are not supported")
        
    def __str__(self):
        return "Node object with value " + str(self.v) + " and derivative " + str(self.d) +\
                " and back-gradient " + str(self.back_g)

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
