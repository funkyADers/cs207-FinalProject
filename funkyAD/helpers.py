import numpy as np
from funkyAD.base import Node

def count_recursive(args):
    '''Counts the number of arguments by recursing over np.arrays and lists'''
    total = 0

    if hasattr(type(args), '__len__'):
        # object is a sequence 
        for x in args:
            total += count_recursive(x)
    else:
        total += 1

    return total

def unpack(args):
    '''Unpacks items in nested np.arrays or lists into a depth-1 list'''
    l = []

    if hasattr(type(args), '__len__'):
        # object is a sequence 
        for x in args:
            l += unpack(x)
    else:
        l.append(args)

    return l

def nodify(args, seed):
    '''Recursively transforms all numerical values in np.arrays and lists into Node objects'''
    i = 0
    new_args = []
    for a in args:
        # TODO: we only support lists and np.arrays at this time
        def agument(x):
            nonlocal i
            node = Node(x, seed[i])
            i += 1
            return node

        if isinstance(a, np.ndarray):
            new_args.append(np.array([agument(x) for x in a]))
            #code below does not work because vectorize can call agument more than len(a) times
            #new_args.append(np.vectorize(agument)(a)) 

        elif isinstance(a, list):
            new_args.append([agument(x) for x in a])

        else:
            new_args.append(agument(a))
    return new_args