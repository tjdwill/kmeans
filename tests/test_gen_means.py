"""
@author: tjdwill
@date: 5 April 2024
@description:
    A test for generating initial means.
"""
import numpy as np
from kmeans.base_funcs import _generate_means as gm


NDIM = 7
v1 = np.eye(NDIM)
v2 = np.zeros((NDIM, NDIM))

# Ensure uniqueness by row.
assert gm(v1, k=NDIM, ndim=len(v1[0])).dtype == np.float64


# Test case with only repeating entries
try:
    gm(v2, k=NDIM, ndim=NDIM)
except ValueError:
    pass


