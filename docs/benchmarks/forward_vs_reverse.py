import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from funkyAD.base import AD

sys.setrecursionlimit(100000)

def my_sum(a):
    return a.sum()

ad_object = AD(my_sum)

# showcase reverse better than forward when inputs increase
fwd_times=[]
rev_times=[]
ninputs = range(1,10000,100)
for n in range(1,10000,100):
    inp = np.array([i for i in range(n)])

    start_time = time.time()
    ad_object._forward(inp)
    runtime = time.time()-start_time
    fwd_times.append(runtime)

    start_time = time.time()
    ad_object._reverse(inp)
    runtime = time.time()-start_time
    rev_times.append(runtime)

# plot results
plt.plot(ninputs, fwd_times, label='forward')
plt.plot(ninputs, rev_times, label='backward')
plt.legend()
plt.title('Comparing times under different AD modes')
plt.xlabel('Number of inputs')
plt.ylabel('Elapsed time in seconds')
plt.savefig('../img/fwd_rev_increasing_n.png')

