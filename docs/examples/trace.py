from funkyAD.base import AD, grad

def f(x, y):
    z = x * y
    return x+y+z

ad = AD(f)._buildtrace(4, 5)
print(ad)