# Milestone1

## Introduction
Differentiation is used in many applications, such as fining stationary points of defined functions or minimizing objective loss functions in machine learning applications. Automatic differentiation (AD) has become one of the most popular techniques for finding derivatives and is often preferred over symbolic differentation and numerical differentiation because of its efficiency and stability.
 
## Background
So how does AD do it? AD takes an input functions and breaks it down into a set elementary functions combined using common mathematical functions, such as addition or multiplication. Then, leveraging the magic of the chain rule, we can calclulate the function's derivatives by calculating the partial derivatives. There are two common methods for implementing AD, forward AD and backward AD (or backpropagation). 

## How to Use funkyAD

The software funkyAD is a software package, that the user will interact with using the AD class. This AD class allows the user to differentiate a specified function using automatic differentiation.

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

```

## Software Organization

#### Directory Structure: 
```
.
|--docs
|  |-base
|  |-utils
|--funkyAD
|  |-tests
|  |-base
|  |-utils
|--examples
 
```
#### Modules (functionality)

The basic package will include a module for forward AD, which takes a function as an argument and
can return any of the following: the derivative, the derivative evaluated at a given value, the trace, etc. We also hope to implement a backward AD module, which takes the same input and returns the same output as the forward AD module, but uses backpropagation to calculate the return values. 

#### Test suite
Our test suite will live within the funkyAD directory. We plan to use TravisCI, CodeCov, doctests and unittests for testing. 

#### Distribution
We will distribute the package with PyPI.  

#### Software packaging: 
[Follow the guidelines of the tutorial.]

## Implementation

In funkyAD we define 3 main classes: AD, Node, and Elementary Function. AD the class that the user will interact with. It takes in a function from the user and calls the necessary functions and classes in order to calculate the gradient. The Node class is essentially a row in our trace table, it has subclasses for input nodes (InputNode) and output nodes (OutputNodes). Nodes are connected and can be added or multiplied together to form new nodes, via the dunder methods __add__ etc. The Elementary Class allows us to define the functions and derivatives of elementary functions passed in by the user, such as sign, log, etc. We also allow the user to add their own elementary function to the list if we do not include the elementary function they need in the initial library list.  

```
**class AD():**

*Methods:*
 
def \_\_init\_\_():  
def \_\_buildgraph\_\_():  
def gradient(forward):  
def createnodes():  
def set\_seed():  
def get\_seed():   
def eval\_trace: return evaluation trace  
def print\_graph: return the evaluation graph   

*Attributes:* 

inputNodeList   
outputNodelist   
eval\_trace   
graph   

**class Node():**    
class InputNode(Node):   
class OutputNode(Node): 

*Methods:*   
\_\_add\_\_  (\_\_radd\_\_)   
\_\_mult\_\_ (\_\_rmult\_\_)   
previous()  
next()  

*Attributes:*   
val   
gradient\_val
 
**class ElementaryFunction():**   
function and its derivative   
ninputs  
noutputs  
add\_function()

*External dependencies:* numpy, doctest, unitest, pytest 
```

How do we deal with arrays ?? 
 linked-list
 graph class? 
 What if we iterate through the list? and AD each in parallel? 
