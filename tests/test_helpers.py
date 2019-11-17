import pytest
import numpy as np
from funkyAD.helpers import count_recursive, unpack, nodify
from funkyAD.base import Node


def test_count_recursive_nparray():
    x = np.array([2,3,1,0])
    assert count_recursive(x)==4

def test_count_recursive_list():
    x = [1,2,3]
    assert count_recursive(x)==3

def test_unpack_ndarray():
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    assert unpack(x)==[1,2,3,4,5,6]

def test_unpack_ndlist():
    x = [[1,2,],[3,4]]
    assert unpack(x)==[1,2,3,4]

def test_nodify_nparray():
    x = np.array([1,2,3])
    seed = [1,2,3]
    assert nodify(x, seed)==[Node(1,1), Node(2,2), Node(3,3)]

def test_nodify_list():
    x = [1,2,3]
    seed = [1,2,3]
    assert nodify(x, seed)==[Node(1,1), Node(2,2), Node(3,3)]

