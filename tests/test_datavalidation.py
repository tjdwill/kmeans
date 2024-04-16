"""
@author: tjdwill
@date: 12 April 2024
@description:
    Testing that each check in the data validation function works
"""
import numpy as np
#-
from kmeans.base_funcs import _validate as val
from kmeans.base_funcs import SMALLEST_THRESH


msg = 'Failed Test\n'

SEED = 27
np.random.seed(SEED)

LEN = 100
valid_data = np.random.random(size=(LEN,1))
K: int = 4

#----------------------k-validation
K_str = '4'

# k is an int
try:
    val(valid_data, k=K_str)
except TypeError:
    pass

# k is within bounds
try:
    val(valid_data, k=0)
except ValueError:
    pass

try:
    val(valid_data, k=LEN+1)
except ValueError:
    pass

#-----------------------tolerance check
try:
    val(valid_data, k=K, tolerance=SMALLEST_THRESH/2)
except ValueError:
    pass

#----------------------max iterations
try:
    val(valid_data, k=K, max_iterations=1.5)
except TypeError:
    pass

try:
    val(valid_data, k=K, max_iterations=-1)
except ValueError:
    pass
    
#--------------------- data check
try:
    val([], k=K)
except ValueError:
    pass

try:
    invalid_data = [arr for arr in valid_data]
    invalid_data.append(np.array([[1, 2], [1, 2]]))
    val(invalid_data, k=K)
except ValueError:
    pass

#----------------------ndim checks

# Type check
ndim_str = '3'
try:
    val(valid_data, k=K, ndim=ndim_str)
except TypeError:
    pass

# Positive value check

try:
    val(valid_data, k=K, ndim=-1)
except ValueError:
    pass


#--------------------------------------initial means check
valid_means = np.array([valid_data[i] for i in range(K)])

# Ensure means and data shape are proper
invalid_means = np.random.random((K, 1, 20))
try:
    val(valid_data, k=K, initial_means=invalid_means)
except AssertionError:
    pass

"""
# A data point that is not among the provided data.
invalid_means = np.copy(valid_means)
invalid_means[-1] = 10 
try:
    val(valid_data, k=K, initial_means=invalid_means)
except ValueError:
    pass
"""
# Duplicate data points
try:
    val(valid_data, k=K, initial_means=[valid_data[0] for _ in range(K)])
except ValueError:
    pass
