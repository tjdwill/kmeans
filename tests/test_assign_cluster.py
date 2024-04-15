"""
@author: tjdwill
@description:
    Test the cluster assignment functions
"""
import numpy as np
from kmeans.base_funcs import _assign_clusters as acs
from kmeans.base_funcs import _generate_means as gm

# Now I need to come up with test data...
SEED = 27
np.random.seed(SEED)

K = 5
NDIM = 3
tst_data = np.random.random(10000).reshape(-1, 1)
means = gm(tst_data, k=K, ndim=NDIM)

# I just learned that Numpy slicing does boundary checks such that--
# for some arr.shape = (1,) for example-- 
# some crazy slice like arr[:404040] will still return the proper value.
# This is why the test still works even though each data element is 1D rather than 3D
"""
Means:

array([[0.06156535],
       [0.47512845],
       [0.98686496],
       [0.97384301],
       [0.40097086]])

tst_data[0]: array([0.42572141])  # Cluster 4
"""
# assert acs(tst_data, means)[0] == 4

clusters = acs(tst_data, means)
assert tst_data[0] in clusters[4] 
print("Cluster Num: Number of Entries")
for key in clusters:
    print(key, len(clusters[key]), sep=": ")

# Test that all elements in cluster 2 are greater than those of all other clusters
test_idx = list(range(K))
test_idx.remove(2)
for idx in test_idx:
    truth_iter = ((clusters[2][i] > clusters[idx]).all() for i in range(len(clusters[2])))
    assert all(truth_iter)

# Test that all elements of cluster 0 are less than those all other clusters
test_idx = list(range(K))
test_idx.remove(0)
for idx in test_idx:
    truth_iter = ((clusters[0][i] < clusters[idx]).all() for i in range(len(clusters[2])))
    assert all(truth_iter)