#==============================================================================
# @author: tjdwill 
# @date: 5 April 2024
# @title: K-means Clustering
# @description:
#     A more function-based implementation of my k-means clustering class.
#==============================================================================
from typing import Union
#-
import numpy as np
#-
from kmeans.base_funcs import _assign_clusters, _validate, _new_centroids, SMALLEST_THRESH


Clusterable = np.ndarray
Clusters = dict[int, Clusterable]

class MaxIterationError(Exception):
    """An exception to be raised when the maximum iteration tolerance is exceeded."""
    pass

def cluster(
        data: Union[Clusterable, list[Clusterable], tuple[Clusterable]],
        k: int,*,
        initial_means: Union[Clusterable, list[Clusterable], tuple[Clusterable]] = None,
        ndim: int = None,
        tolerance: float = SMALLEST_THRESH, 
        max_iterations: int = 250,
) -> tuple[Clusters, Clusterable]:
    """Perform k-means clustering

    The input data should be formatted in terms of row vectors. Given a flat
    numpy array ``data=np.array([0, 1, 2, 3, 4])``, do the following::

        data = data.reshape(data.shape[-1], -1)
        # or 
        data = data[..., np.newaxis]

    It should make each point a row entry::

        [[0], [1], [2], [3], [4]]
    
    Data of higher dimensions (ex. a multi-channeled image)
    should be flattened using the number of indices
    for the deepest dimension. So, for an image with shape
    (480, 640, 3), run::

        data = data.reshape(-1, data.shape[-1])

    Args:
        data: The input data. Expects data homogeneity (all elements are the
            same dimension)
        k: Amount of clusters desired.
        initial_means: The initial cluster centroids. Means are randomly
            selected from data with uniform probability by default.
        ndim: Dimension limit for clustering. If default, the length of a given
            data element is used (all data dimensions clustered).
        tolerance: Controls the completion criteria. Lower values -> more
            iterations. Defaults to 20*eps for np.float64.
        max_iterations: Max number of iterations before terminating function execution.

    Returns:
        dict[int, np.ndarray], np.ndarray: Clustered Data, Cluster Centroids

    Raises:
        kmeans.MaxIterationError: Raise this exception if the clustering doesn't
            converge before reaching the `max_iterations` count.

    """
    data, initial_means, ndim = _validate(
        data,
        k, 
        initial_means=initial_means, 
        ndim=ndim,
        tolerance=tolerance, 
        max_iterations=max_iterations
    )
    clusters, old_centroids = {}, initial_means

    for _ in range(max_iterations):
        clusters = _assign_clusters(data, old_centroids)
        centroids = _new_centroids(clusters, ndim)
        changes = np.linalg.norm(centroids - old_centroids, axis=1)  # Distance along each vector
        if any(np.where(changes > tolerance, True, False)):
            old_centroids = centroids
        else:
            return clusters, centroids
    else:
        raise MaxIterationError("Iteration count exceeded.")            
