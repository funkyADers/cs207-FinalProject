from funkyAD.base import AD, grad


def cube(x):
    return x ** 3

print(grad(cube)(10))
print(AD(cube)._reverse(10))


from funkyAD.functions import exp

def exponential(x):
    return exp(x)

print(grad(exponential)(1))
print(AD(exponential)._reverse(1))

import numpy as np

#a = np.array([1])

#print(a.__class__([4, 5]))


def add(a):
    s = 0
    for x in a:
        s += x
    return s

print(grad(add)(np.array([4, 1, 8])))
print(AD(add)._reverse(np.array([4, 1, 8])))


def add2(x):
    return x.sum()

print(grad(add2)(np.array([2, 3])))
print(AD(add2)._reverse(np.array([2, 3])))


def pair(x, y):
    return x,y

print(grad(pair)(5, 2))
print(AD(pair)._reverse(5, 2))


def minimum(x, y):
    if x < y:
        return x
    else:
        return y

print(grad(minimum)(5, 2))
print(AD(minimum)._reverse(5, 2))
