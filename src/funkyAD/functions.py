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
        new.f = self
        new.parents = new_args

        return new

addition = BaseFunction(lambda x, y: x.v + y.v, lambda x, y: x.d + y.d)
subtraction = BaseFunction(lambda x, y: x.v - y.v, lambda x, y: x.d - y.d)

# Product rule
multiplication = BaseFunction(lambda x, y: x.v * y.v, lambda x, y: x.d * y.v + x.v * y.d)

# Quotient rule
division = BaseFunction(lambda x, y: x.v / y.v, lambda x, y: (y.v * x.d - x.v * y.d) / (y.v ** 2))

# Generalized Power Rule
power = BaseFunction(lambda x, y: x.v ** y.v,
                     lambda x, y: (x.v ** y.v) * (x.d * y.v / x.v + y.d * np.log(x.v)) )

sqrt = BaseFunction(lambda x: x.v**0.5, lambda x: x.d / (2*x.v**0.5))

pos = BaseFunction(lambda x: +x.v, lambda x: +x.d)
# Negation (or taking the negative of, i.e. changing sign)
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

# Rounding the value of a node with specified digits
def round1(x, n=0):
    from .base import Node
    if isinstance(n, Node):
        n = n.v
    return np.round(x.v * (10 ** n)) / (10. ** n)

# Roundin the derivative of a node with specified digits
def round2(x, n=0):
    from .base import Node
    if isinstance(n, Node):
        n = n.v
    return r_der(x.v * (10 ** n))

# Rounding
_round = BaseFunction(round1, round2)
floor = BaseFunction(lambda x: np.floor(x.v), lambda x: r_der(x.v))
ceil = BaseFunction(lambda x: np.ceil(x.v), lambda x: r_der(x.v))
trunc = BaseFunction(lambda x: np.trunc(x.v), lambda x: r_der(x.v))

floordiv = BaseFunction(lambda x, y: x.v // y.v, lambda x, y: r_der(x.v / y.v))

# np.e**2 != np.exp(2), the former is slightly more precise
# math.e**2 != np.exp(2), same
# Most applications of exponents use np.exp(x) rather than np.e**x,
# so for them to give expected results (e.g. tests on values) we
# need to use np.exp() when applicable, and manual ** otherwise

# Decorator to verify valid base of exponentiation or logarithm
def base_check(f):
    def inner(x, b = np.e):
        from .base import Node
        if isinstance(b, Node):
            b = b.v
        if b <= 0:
            raise ValueError('Base must be positive')
        return f(x, b)
    return inner

# Value of exponentials
@base_check
def exp1(x, b = np.e):
    # np.exp(2) != np.e**2 --> latter more precise
    if b == np.e:
        # Hence, for consistency with usage of np.exp():
        return np.exp(x.v)
    else:
        return b ** x.v

# Derivative of exponentials
@base_check
def exp2(x, b = np.e):
    if b == np.e:
        return x.d * np.exp(x.v)
    else:
        return x.d * np.log(b) * b ** x.v

exp = BaseFunction(exp1, exp2)

@base_check
def log1(x, b = np.e):
    # np.log(np.e) == 1
    return np.log(x.v) / np.log(b)

@base_check
def log2(x, b = np.e):
    return x.d / (x.v * np.log(b))

log = BaseFunction(log1, log2)

# Trigonometric functions
sin = BaseFunction(lambda x: np.sin(x.v), lambda x: x.d * np.cos(x.v))
cos = BaseFunction(lambda x: np.cos(x.v), lambda x: -x.d * np.sin(x.v))
tan = BaseFunction(lambda x: np.tan(x.v), lambda x: x.d / (np.cos(x.v) ** 2))

# Complex numbers are not supported: verify inv angle in valid range
def one_check(f):
    def inner(x):
        from .base import Node
        if isinstance(x, Node):
            val = x.v
        else:
            val = x
        if not np.abs(val) < 1:
            raise ValueError('Angle must be between -1 and 1 exclusive')
        return f(x)
    return inner

arcsin = BaseFunction(
    one_check(lambda x: np.arcsin(x.v)), 
    one_check(lambda x: x.d/np.sqrt(1 - x.v**2))
)
arccos = BaseFunction(
    one_check(lambda x: np.arccos(x.v)), 
    one_check(lambda x: -x.d / np.sqrt(1 - x.v**2))
)
arctan = BaseFunction(
    one_check(lambda x: np.arctan(x.v)), 
    one_check(lambda x: x.d / (1 + x.v**2))
)

# Hyperbolic functions
# Note: can implement manually using natural exponential, but similarly to before
# using np.exp() vs Numpy's hyperbolic functions directly give very slightly
# different results in the 14th or so decimal, so == comparisons yield false
# E.g. manual sinh = 3.626860407847019, while np.sinh = 3.6268604078470186
# Since most users presumably use np.sinh or similar, we implement using that
# Otherwise their tests may fail

sinh = BaseFunction(lambda x: np.sinh(x.v), lambda x: x.d*np.cosh(x.v))
cosh = BaseFunction(lambda x: np.cosh(x.v), lambda x: x.d*np.sinh(x.v))
tanh = BaseFunction(lambda x: np.tanh(x.v), lambda x: x.d/(np.cosh(x.v)**2))

# General logistic function
# Many ways of defining it, we do it using L(x) = upper_bound/(1+rate*exp(-x))
logistic = BaseFunction(
    lambda x, u, r: u.v/(1+r.v*np.exp(-x.v)), 
    lambda x, u, r: x.d*u.v*r.v*np.exp(x.v)/(r.v + np.exp(x.v))**2
)

# Sigmoid: specific case of logistic used for neural nets
sigmoid = BaseFunction(
    lambda x: 1/(1+np.exp(-x.v)), lambda x: x.d*np.exp(x.v)/(1+np.exp(x.v))**2)
