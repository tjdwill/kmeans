"""
@author: tjdwill 
@date: 5 April 2024
@title: K-means Clustering
@description:
    A more function-based implementation of my k-means clustering class.
"""

import itertools
from functools import partial
from typing import Union, List, Tuple, Callable
import numpy as np
from numpy.typing import NDArray


Clusterable = Union[List[NDArray], Tuple[NDArray], NDArray]
Clusters = dict[int: Clusterable]

# A la Peter Corke's spatialmath, this sets the smallest value in which an element can change. 
_eps = np.finfo(np.float64).eps
SMALLEST_THRESH =  20*_eps


class MaxIterationError(Exception):
    """An exception to be raised when the maximum iteration threshold is exceeded."""
    pass


def _eucl_dist(x1: NDArray, x2: NDArray, ndim: int) -> np.float64:
    """Calculates Euclidian Distance
    
    Args:
        x1: The first data point
        x2: The second data point
        ndim: How many dimensions we consider
    
    Returns:
        np.float64: the calculated distance
    """
    # Assume data satisfies ndim
    # Assume flat data.
    # Assume ndim >= 1
    vec = x1[:ndim] - x2[:ndim]
    return np.linalg.norm(vec).astype(np.float64)


# TODO: Add more distance functions (efficiency)
# Function mappings
dist_funcs = {
    "euclidean": _eucl_dist,
}


def _assign_cluster(
        data: NDArray, centroids: NDArray, *,
        dist_func: Callable, ndim: int, k: int
) -> int:
    """Assigns a given data element to a cluster.
        
    Args:
        data: The data to be labeled.
        centroids: The given information to use as cluster criteria.
        dist_func: The method of calculating distance.
        ndim: Data dimensions
        k: Number of centroids
    
    Returns:
        int: The label index
    """
    # Get the smallest distance among the centroids
    x1s = (data,)*k

    distances = np.array(list(map(dist_func, x1s, centroids, (ndim,)*k)))
    label: tuple[NDArray] = np.nonzero(
        np.where(distances == min(distances), True, False)
    )
    idx: int = label[0][0]
    return idx


def _assign_clusters(
        data: Clusterable, centroids: NDArray, dist_func: Callable
) -> Clusters:
    """Place each data point into a cluster

    Args:
        data: The data to be labeled.
        centroids: The given information to use as cluster criteria.
        dist_func: The method of calculating distance

    Returns:
        dict[int: Clusterable]: the clusters 
    """
    k = len(centroids)
    ndim = centroids[0].shape[0]
    clusters = {i: [] for i in range(k)}

    partial_assign = partial(_assign_cluster, dist_func=dist_func, k=k, ndim=ndim)
    labels = map(partial_assign, data, (centroids,)*len(data))

    groupings = zip(labels, data)  # Internal Element Format: (label, data)
    # Is there a way to extract the groupings all at once?
    for key, item in groupings:
        clusters[key].append(item)
    
    return clusters


def _generate_means(data: Clusterable, k: int, ndim:int) -> NDArray:
    """Randomly selects initial means with uniform distribution

    Args:
        data: The data from which the means are selected
        k: How many means to select
        ndim: Dimensionality of means

    Returns:
        NDArray: the initial means    

    Raises:
        ValueError: If can't find unique set of means within the COUNT_OUT
    """
    COUNT_OUT = 1000
    count = 0

    while count < COUNT_OUT:
        indices = np.random.choice(np.arange(len(data)), size=(k,), replace=False) 
        means = [data[i, :ndim] for i in indices]
        if len(np.unique(means, axis=0)) == len(means):
            return np.array(means)
        count += 1
    else:
        raise ValueError("Could not find unique set of initial means.\n")


def _new_centroids(clusters: Clusters, ndim: int) -> NDArray:
    """Returns a new set of centroids
    
    Args:
        clusters: the current grouped data
        ndim: Dimension of data we are clustering

    Returns:
        NDArray: the new centroids
    """
    centroids = []
    for key in clusters:
        cluster = clusters[key]
        clust_arr = np.array([arr[:ndim] for arr in cluster])
        avg = clust_arr.sum(0) / clust_arr.shape[0]
        centroids.append(avg)
    else:
        return np.array(centroids)


def _validate(
        data: Clusterable,
        k: int,*,
        initial_means: Clusterable = None,
        ndim: int = None,
        threshold: float = 0.5,
        max_iterations: int = 100,
) -> tuple[Clusterable, NDArray, int]:
    """Perform validation checks on cluster arguments
    
    Args:
        data: The input data
        k: Amount of clusters
        initial_means: The initial cluster centroids
        ndim: Dimension limit for clustering
        threshold: How much can a given cluster centroid 
            move between iterations
        max_iterations: The counter timeout

    Returns:
        tuple[Clusterable, NDArray, int]: the validated data, initial means, and ndim
    """
    # Check k
    if not isinstance(k, int): 
        raise TypeError("k must be an integer.")
    if k < 1 or k > len(data):
        raise ValueError("k must be positive and can't exceed number of data points.")
    
    # Check threshold
    if threshold < SMALLEST_THRESH:
        raise ValueError("Threshold too small.")
    
    # Check max_iterations
    if not isinstance(max_iterations, int):
        raise TypeError("Max iterations must be an integer.")
    if max_iterations < 0:
        raise ValueError("Max iterations must be greater than 0.")

    # Check data
    if not len(data):
        raise ValueError("No data provided.")
    try:
        assert all(arr.ndim == 1 for arr in data)
    except AssertionError:
        raise ValueError("Each data entry must be a row vector (meaning of shape <1, m>)")

    # Check ndim
    if ndim is None:
        ndim = min(x.shape[0] for x in data)
    elif not isinstance(ndim, int):
        raise TypeError("Dimension parameter must be an integer.")
    elif ndim < 0:
        raise ValueError("Dimension parameter must be positive.")
    elif ndim > min(x.shape[0] for x in data):
        raise ValueError("Dimension value cannot exceed data dimensionality.")
    else:
        pass

    # Check initial means
    if initial_means is None:
        temp_means = _generate_means(data, k=k, ndim=ndim)
    else:
        # Create homogeneous data; Each entry is a row vector.
        temp_means = np.array([arr[:ndim] for arr in initial_means])
        temp_data = np.array([arr[:ndim] for arr in data])
        assert temp_means.shape[-1] == temp_data.shape[-1]
        # check if all elements in the means are in the data
        for centroid in temp_means:
            # Test if the centroid is one of the data entries in temp_data.
            try:
                assert any(np.equal(centroid, temp_data).all(1))
            except AssertionError:
                raise ValueError(f"Initial means item not among provided data: {centroid}")
        
        # Check for duplicates
        filtered_initial_means, ndx = np.unique(
            np.copy(temp_means),  # use copy so original array order remains the same.
            axis=0,
            return_index=True
        )
        # Check length
        try:
            assert len(filtered_initial_means) == k
        except AssertionError:
            raise ValueError("\nNumber of unique mean points must == number of segments.\n")
    return data, temp_means, ndim


def cluster(
        data: Union[List[NDArray], Tuple[NDArray], NDArray],
        k: int,*,
        initial_means: Union[List[NDArray], Tuple[NDArray], NDArray] = None,
        ndim: int = None,
        threshold: float = 0.5,
        max_iterations: int = 100,
        dist_func: Callable = "euclidean"
):
    """Perform k-means clustering
    
    Args:
        data: The input data
            This data should be formatted in terms of row vectors.
            In other words, given a flat numpy array
            data=np.array([0, 1, 2, 3, 4]), do the following:
            `data = data.reshape(data.shape[-1], -1)`
            or `data = data[..., np.newaxis]`
            It should make each point a row entry:
                [[0], [1], [2], [3], [4]]
            Pass this version into the KMeans constructor.
            Data of higher dimensions (ex. a multi-channeled image)
            should be flattened using the number of indices
            for the deepest dimension. So, for an image with shape
            (480, 640, 3), run
                `data = data.reshape(-1, data.shape[-1])`
            and pass this data into the constructor.
        k: Amount of clusters
        initial_means: The initial cluster centroids
            Defaults to `None` -> Means are randomly selected from data 
            with uniform probability
        ndim: Dimension limit for clustering; 
            Defaults to None -> selects the ndim based on data dimensionality
        threshold: How much can a given cluster centroid 
            change between iterations. Default: 0.5
        max_iterations: The counter timeout 
            Default: 100
        dist_func: Distance calculation method
            Default: 'euclidean'

    Returns:
        tuple[Clusterable, NDArray, int]: the validated data, initial means, and ndim

    Raises:
        MaxIterationError: Raise this exception if the clustering doesn't converge before
            reaching the `max_iterations` count.
    """
    data, initial_means, ndim = _validate(
        data,
        k, 
        initial_means=initial_means, 
        ndim=ndim,
        threshold=threshold, 
        max_iterations=max_iterations
    )

    clusters, old_centroids = {}, initial_means

    for _ in range(max_iterations):
        clusters = _assign_clusters(data, old_centroids, dist_funcs[dist_func])
        centroids = _new_centroids(clusters, ndim)
        changes = np.linalg.norm(centroids - old_centroids, axis=1)  # Distance along each vector
        if any(np.where(changes > threshold, True, False)):
            old_centroids = centroids
        else:
            return clusters, centroids
    else:
        raise MaxIterationError("Iteration count exceeded.")            
