import numpy as np


class AD():

    def __init__(self, f):
        self.f = f
        self.seed = None
        self.n = None
        self.m = None

    def set_seed(self, seed):
        self.seed = seed


    def _check_seed(self, l):
        # Checks if seed has appropriate dimension
        if len(seed) != l:
            raise ValueError("Seed dimension does not match input dimension")

    def _evaluate(self, *args):
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

    def __init__(self, v, d=0):
        self.v = v
        self.d = d # If derivative is None it's assumed to be 0 (for a constant node)
        
        #self.prev = []
        #self.next = []

    def __str__(self):
        return "Node object with value " + str(self.v) + " and derivative " + str(self.d)

    def __add__(self, other):
        return addition(self, other)

    def __radd__(self, other):
        return addition(self, other)

    def __mul__(self, other):
        return multiplication(self, other)

    def __rmul__(self, other):
        return multiplication(self, other)

    def __pow__(self, other):
        return power(self, other)

def grad(f):
    # Syntactic sugar for AD(f).grad
    return AD(f).grad

# Import statement has to be at the bottom for some reason
from functions import addition, multiplication, power
from helpers import count_recursive, nodify

