import numpy as np
from kmeans import view_clustering


SEED = 27
np.random.seed(SEED)
NDIM = 2 
K = 5
# Each data point has five entries to show the `ndim` parameter works
data = np.random.random((20000, 5))

clusters, centroids, fig = view_clustering(data, k=K, ndim=NDIM)