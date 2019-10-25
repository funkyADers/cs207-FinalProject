# Milestone1

## Introduction
Differentiation is used in many applications, such as finding stationary points of defined functions or minimizing objective loss functions in machine learning applications. But differentiating an arbitrary function $\mathbb{R}^n \rightarrow \mathbb{R}^m$ is generally not an easy task. In case the function can be expressed as a composition of differentiable elementary functions (which in most cases is true), Automatic Differentiation can help. AD has become one of the most popular techniques for finding derivatives and is often preferred over symbolic differentation and numerical differentiation because of its efficiency and stability.
 
## Background
So how does AD do it? AD takes an input functions and breaks it down into a set of elementary functions combined using common mathematical functions, such as addition or multiplication. Then, leveraging the magic of the chain rule, the function's derivatives are calculated using the partial derivatives w.r.t. the inputs. Basically you compute an evaluation trace (which can be stored in either a table or a graph), where at each intermediate step of the computation you store the current value(s) of the intermediate variables and their derivatives w.r.t. some input seed vectors. 

As an example, if you want to compute the derivative of $\sin(\tan(xy) + \cos(x + y))$ you can first compute the derivatives w.r.t $x,y$ of $\tan(xy)$ and $\cos(x + y)$, then add those together, and then get the derivative of the entire function using the chain rule. This is not done symbolically, but rather numerically, for every given input.

There are two common methods for implementing AD, forward AD and backward AD (of which the popular backpropagation algorithm for neural networks is a special case), which differ in efficiency based on the dimension of input/outputs. 

## How to Use funkyAD

The software funkyAD is a software package that the user will interact with using the AD class. This AD class allows the user to differentiate a specified function by wrapping it into an AD object, automatically differentiate it, access the results and inspect the intermediate steps (if desired).

The package is intended for use by developers on personal computers, as a building block on top of which other functionality may be developed and depend.

Pseudocode on how to interact with funkyAD shown below: 
```
from funkyAD import AD

# define a function to differentiate 
def f(x):
   return 2*x+3
   
# Option 1: directly get the gradient or output 
AD(f).gradient(5) # outputs 10

# Option 2: create an AD object, and interact with it as you wish, 
#  e.g. obtain the gradient at a specified value
#  e.g. obtain the evaluation trace
obj = AD(f)
obj.gradient(5) # outputs 10
obj.eval_trace() # outputs the evaluation table

# Option 3: syntactic sugar for simpler syntax
from funkyAD import gradient
gradient(f)(5) # outputs 10
# funkyAD accepts functions with multiple inputs
def f2(x,y):
    return 2x+y
def f3(a -> np.array):
	return a[1] + a[2]

obj2 = AD(f2).gradient(4, 5)
obj3 = AD(f3).gradient(np.array([3, 4, 5]))

```

## Software Organization

#### Directory Structure: 
```
.
|--docs
|  |-base
|  |-utils
|
|--funkyAD
|  |-tests
|  |-base
|  |-utils
|
|--examples
|
|--benchmarks
```
#### Modules (functionality)

The basic package will include a module for forward AD, which takes a function as an argument and
can return any of the following: the derivative, the derivative evaluated at a given value, the trace, etc. We also hope to implement a backward AD module, which takes the same input and returns the same output as the forward AD module, but uses backpropagation to calculate the return values and is more efficient for single-output functions.

#### Test suite
Our test suite will live within the funkyAD directory. We plan to use TravisCI, CodeCov, doctests and unittests for testing. 

We plan to write extensive and detailed docstrings for all function that will be accessible to the user, and replicate those in a nicer format in the corresponding docs folder (e.g. the file /funkyAD/AD.py will have its docs in /funkyAD/AD.html). A nice collection of examples for installation and usage will hopefully help the user in getting a working knowledge of the library quickly.

We plan to use unittests for most of our testing (dottests only occasionally). We aim for as much code coverage as possible, and specifically target Exception raising and handling of edge cases.

#### Distribution
The funkyAD package will be distributed with PyPI. Consequently, users will be able to install the package using the uniquitous pip package manager. It will follow the guidelines and instructions in the official Python documentation.

#### Software packaging: 
According to common practice, we will include \_\_init\_\_.py files and setup.py so that automated tools can install and set up the library properly. We will choose transparent file names so that import statements are intuitive to the end user.

It will be packaged as a Wheel for fast and easy installation.

## Implementation

In funkyAD we define 3 main classes: AD, Node, and ElementaryFunction. AD is the class that the user will interact with. It takes in an arbitrary function from the user and calls the necessary functions and classes in order to calculate the gradient. It will have transparently named methods and syntactic sugar when appropriate. The user should not know how anything else in the library works for simple usage.

The Node class is essentially a row in our trace table, it has subclasses for input nodes (InputNode) and output nodes (OutputNodes). Nodes are connected and can be added or multiplied together to form new nodes, via the dunder methods\_\_add\_\_ etc, which allows us to build up the trace table. A Node is, essentially, an extension of the Box class in HW4. If the user has defined a function f, we will call f(InputNode(input)) and the successive operation performed on that node will allow us to recover the evaluation trace.

The ElementaryFunction Class defines the functions and derivatives of elementary functions passed in by the user, such as sin, log, etc. We also allow the user to add their own elementary function to the list if we do not include the elementary function they need in the initial library list. An appropriate exception is raised if the user tries to utilize a function that is not defined as an instance of the ElementaryFunction class.

```
# External dependencies
import numpy
import doctest
import unitest
import pytest

# the AD class instigates the differentation process and stores all output values
class AD():

Methods:
 
def __init__(): initialize AD object 
def _buildgraph(): build the evaluation graph
def gradient(forward): run forward AD, allows for option for backward AD if implemented
def _createnodes(): create notes for the evaluation trace
def set_seed(): set the seed (by default is the identity matrix)
def get_seed(): return the seed
def eval_trace: return evaluation trace  
def print_graph(): return the evaluation graph   

Attributes: 

nodeList: list of nodes
inputNodeList: the list of inputs
outputNodelist: the list of outputs
eval_trace: the evaluation trace   
graph: the evaluation graph 
history: history of calls memoized for increased performance
(dunder methods e.g. __str__ redefined)

# Node classes and subclasses 
class Node(): the node class stores relevant information for each node in the graph
class InputNode(Node): a subclass of the Node class
class OutputNode(Node): a subclass of the Node class

Methods:
__add__  (__radd__): add two nodes
__mult__ (__rmult__): multiple two nodes
(other dudner methods redefined as appropriate)

Attributes:   
val: the value of the node   
gradient_val: the gradient of the node
parents: list of parent nodes
children: list of children nodes
el: elementary function that the node computes (an ElementaryFunction object)
SHOULD THIS BE STORED AS A DUAL NUMBER?

# The Elementary Function stores elementary functions and their derivatives
class ElementaryFunction(): 

Methods:
__call__(): returns the output of the function
derivative(): returns the derivative for a given input

Attributes: 
ninputs: number of inputs  
noutputs: number of outputs 


```

**OUTSTANDING**
Dual numbers are essentially encoded as the pair (Node.val, Node.grad_val) in our implementation, and additions/operation are performed when calling ElementaryFunction(Node1, Node2).

At this point we do not know how to deal with arbitrary-length arrays. One option might be to subclass np.array so it includes the functionalities we need. Another option is to have the user declare the length of the array they are passing to the function so we can create an appropriate amount of InputNodes and OutputNodes.
