from funkyAD.base import AD, grad

def f(x, y):
    return x * y

ad = AD(f)._reverse(4, 5)
print(ad)

print(AD(f).grad(4, 5))