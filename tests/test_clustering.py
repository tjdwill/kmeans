"""
@author: tjdwill
@date: 6 April 2024
@description:
    Test the k-means clustering function
"""

import numpy as np
from kmeans import cluster
from kmeans.base_funcs import _generate_means as gm

SEED=27
np.random.seed(SEED)
SZ = 100000
NDIM = 3
K = 4

data = np.random.random((SZ,10))
means = gm(data, K, NDIM)
print("Means:\n", means)
clusters, centroids = cluster(data, k=K, ndim=NDIM, initial_means=means)

print("Centroids:\n", centroids)
for key in clusters:
    print(key, len(clusters[key]), sep=": ")
