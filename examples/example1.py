# This import statement does not work
from funkyAD.base import AD


def add(x, y):
    return x + y + 2

print(AD(add).evaluate(4, 5))


def mult(x, y):
    return x * y * 1

print(AD(mult).evaluate(4, 5))


def cube(x):
    return x ** 3

print(AD(cube).evaluate(10))


from functions import exp

def exponential(x):
    return exp(x)

print(AD(exponential).evaluate(1))