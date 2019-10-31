from base import AD, grad


def cube(x):
    return x ** 3

print(AD(cube)._evaluate(10))


from functions import exp

def exponential(x):
    return exp(x)

print(AD(exponential)._evaluate(1))


import numpy as np

#a = np.array([1])

#print(a.__class__([4, 5]))


def add(a):
    s = 0
    for x in a:
        s += x
    return s

print(AD(add)._evaluate(np.array([4, 1, 8])))

def add2(x):
    return x.sum()

print(AD(add2)._evaluate(np.array([2, 3])))

def pair(x, y):
    return x,y

print(AD(pair)._evaluate(5, 2))
