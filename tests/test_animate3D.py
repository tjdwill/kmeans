"""
@author: tjdwill
@date: 8 April 2024
@description:
    Testing 3-D plots
"""
import numpy as np
from kmeans import view_clustering


SEED = 27
np.random.seed(SEED)
NDIM = 3
K = 4
# Each data point has five entries to show the `ndim` parameter works
data = np.random.random((20000, 5))

clusters, centroids, fig = view_clustering(data, k=K, ndim=NDIM, tolerance=0.001, max_iterations=200)