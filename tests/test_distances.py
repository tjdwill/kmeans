"""
@author: tjdwill
@date: 5 April 2024
@description:
    A test for the distance calculation function(s).
    Tests both value and, implicitly, dimension specification.
"""
import numpy as np
from kmeans.base_funcs import _eucl_dist


NDIM = 3
MAX_DIM = 10

triangle = np.arange(3, MAX_DIM)
zero_case = np.random.randint(0, 2**32 - 1, size=(MAX_DIM,), dtype=np.uint32)
zeros = np.zeros(MAX_DIM)
identity = np.zeros(NDIM)
identity[0] += 1

assert _eucl_dist(triangle, zeros, 2) == 5                  # ||<3, 4>|| == 5
assert _eucl_dist(zero_case, zero_case, MAX_DIM) == 0       # ||v - v|| == 0
assert _eucl_dist(identity, zeros, NDIM) == identity[0]     # ||v - \vec{0}|| == ||v||
