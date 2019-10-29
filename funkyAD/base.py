from inspect import signature
import numpy as np


class AD():

    def __init__(self, f):
        self.f = f
        self.n = len(signature(self.f).parameters) # number of inputs
        self.seed = np.eye(self.n)

    def evaluate(self, *args):
        return self.f(*[Node(val, self.seed[i]) for i, val in enumerate(args)])


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

# Import statement has to be at the bottom for some reason
from functions import addition, multiplication, power

