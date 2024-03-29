import numpy as np

def count_recursive(args):
    '''Counts the number of arguments by recursing over np.arrays and lists'''
    #deferred import to work around circular dependency
    from .base import Node
    if isinstance(args, (np.ndarray, list, tuple)):
        pass
    elif isinstance(args, (int, float, Node, np.int64)):
        return 1
    else:
        raise TypeError('The input argument should be either np.arrays or list')
    total = count_recursive_recursion_part(args)
    return total
    
def count_recursive_recursion_part(args):
    total = 0
    if hasattr(type(args), '__len__'):
        # object is a sequence

        for x in args:
            total += count_recursive_recursion_part(x)
    else:
        total += 1
    return total

def unpack(args):
    '''Unpacks items in nested np.arrays or lists into a depth-1 list'''
    #deferred import to work around circular dependency
    from .base import Node
    if isinstance(args, (np.ndarray, list, Node)):
        pass
    else:
        raise TypeError('The input argument should be either np.arrays or list')
    
    l = []
      
    if hasattr(type(args), '__len__'):
        # object is a sequence
        for x in args:
            l += unpack_recursion_part(x)
    else:
        l.append(args)

    return l
    
def unpack_recursion_part(args):
    l = []
      
    if hasattr(type(args), '__len__'):
        # object is a sequence
        for x in args:
            l += unpack_recursion_part(x)
    else:
        l.append(args)

    return l

def nodify(args, seed):
    '''Recursively transforms all numerical values in np.arrays and lists into Node objects'''
    if isinstance(args, (np.ndarray, list, tuple)):
        pass
    else:
        raise TypeError('The input argument should be either np.arrays or list')
    i = 0
    new_args = []
    for a in args:
        # TODO: we only support lists and np.arrays at this time
        def agument(x):
            # Deferred import to work around circular dependencies
            from .base import Node
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

def recursive_append(n, trace):
    # appends a node and all its parents to trace by recursing
    trace.append(n)
    if n.parents is not None:
        for p in n.parents:
            recursive_append(p, trace)

