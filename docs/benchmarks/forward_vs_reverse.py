import time
import sys
import numpy as np
from funkyAD.base import AD

sys.setrecursionlimit(100000)

def my_sum(a):
    return a.sum()

ad_object = AD(my_sum)
inp = np.array([n for n in range(10000)])

start_time = time.time()
print("Forward mode call")
ad_object._forward(inp)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("Reverse mode call")
ad_object._reverse(inp)
print("--- %s seconds ---" % (time.time() - start_time))
