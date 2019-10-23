# Milestone1
### Project Group 4

## Introduction
Differentiation is used in 

## Background
How AD works

## How to Use funkyAD

The software funkyAD is a software package that includes 3 important classes: AD, Node, and Elementary Function.
The user will interact with the AD class, which performs autodifferentaiotn 

Pseudocode on how to interact with funkyAD shown below: 

```
from funkyAD import AD

# define a function to differentiate 
def f(x):
   return 2*x+3
   
# Option 1: directly get the gradient or output 
AD(f).gradient()

# Option 2: create an AD object, and interact with it as you wish, 
#  e.g. obtain the gradient at a specified value
#  e.g. obtain the evaluation trace
obj = AD(f)
obj.gradient(5)
obj.eval_trace()

# Option 3: syntactic sugar for simpler syntax
from funkyAD import gradient
gradient(f)(5)

# funkyAD accepts functions with multiple inputs
def f(x,y):
    return 2x+y

obj2 = AD(f)

# user adds their own base function

```

## Software Organization

#### Directory Structure: 
.\docs
    \base 
    \utils
.\funkyAD
    \tests
    \base
    \utils
.\examples

#### Modules (functionality)
forward - forward diffferentiation AD, give function as an argument and return
backward - backward differentation AD

#### Test suite
Our test suite will live with \funkyAD
TravisCI, CodeCov, and both doctests and unittests

#### Distribution
PyPI  

#### Software packaging: 
Follow the guidelines of the tutorial 

## Implementation

\bold{class AD():}

Methods: 
def __init__():
def __buildgraph__():
def gradient(forward):
def createnodes():
def set_seed():
def get_seed(): 
def eval_trace: return evaluation trace
ddef print_graph: return the evaluation graph 

Attributes: 
inputNodeList
outputNodelist
eval_trace
graph

\bold{class Node():}
\bold{class InputNode(Node):}
\bold{class OutputNode(Node): }

Methods:
 __add__  (__radd__)
 __mult__ (__mult__)
 previous()
 next()
 
 Attributes:
 val
 gradient_val
 

# handle derivates of elementary functions (e.g. sin, sqrt)
\bold{class ElementaryFunction(): }
 function and its derivative
 n_inputs
 n_outputs
 add_function()


External dependencies: numpy, doctest, unitest, pytest 

How do we deal with arrays ?? 
 linked-list
 graph class? 
 What if we iterate through the list? and AD each in parallel? 
