#==============================================================================
# @author: tjdwill 
# @date: 6 April 2024
# @title: k-means base functions
# @description:
#     Functions that assist with the clustering operation
#==============================================================================
from time import perf_counter
from typing import Callable, Union
#-
import numpy as np

Clusterable = np.ndarray 
Clusters = dict[int, Clusterable]

# A la Peter Corke's spatialmath, this sets the smallest value in which an element can change. 
_eps = np.finfo(np.float64).eps
SMALLEST_THRESH =  20*_eps


def time_func(func: Callable) -> Callable:
    """A decorator to time function execution.
    
    Args:
        func: the function to time.

    Returns: 
        Callable: The wrapped function

    """
    def wrapper(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        print(f"{func.__name__} Execution Time (s): {perf_counter() - start}")
        return out
    return wrapper


#@time_func 
def _assign_clusters(data: Clusterable, centroids: np.ndarray) -> Clusters:
    """Assigns each data element to a cluster
    
    Args:
        data: The data to be labeled.
        centroids: The given information to use as cluster criteria.

    Returns:
        dict[int, np.ndarray]: The Clusters 
    
    """    
    k, ndim, *_ = centroids.shape
    temp_data = data[..., :ndim]
    vecs = temp_data[:, np.newaxis] - centroids[np.newaxis, ...]
    norms = np.linalg.norm(vecs, axis=-1)

    # Handle the case in which a given data point is equivalently close to multiple centroids.
    '''
    Step 1: For each given set of norms, find where the elements are equal to the minimum of that set.
        Ex. [12, 2, 3, 2] --> [False, True, False, True]
        eq = np.equal(norms, np.min(norms, axis=1)[:, np.newaxis])
    Step 2: Find where the elements are True using `np.nonzero`  (call the resulting structure `p`). 
        This gives the indices of each True value: 
        `p = np.nonzero(eq)  # (y_indices: np.ndarray, x_indices: np.ndarray)`
    Step 3: For the y_indices (the top level indices), call `np.unique` and return the indices needed to perform the operation. 
        _, idxs = np.unique(p[0], return_index=True)
    Step 4: Now, generate the y and x indices needed to construct the array with only one nonzero term in each element:
        y = p[0][idxs]
        x = p[1][idxs]
    Step 5: Create a zero matrix and fill it with ones in the proper place
        labels = np.zeros(norms.shape)
        labels[y, x] = 1
    Step 6: Now, we can find the nonzero elements of this structure and proceed
        to find the label values for each data element. 
    '''
    broadcastable = np.min(norms, axis=1)[:, np.newaxis]
    nz = np.nonzero(np.equal(norms, broadcastable))
    _, idxs = np.unique(nz, return_index=True)
    y = nz[0][idxs]
    x = nz[1][idxs]
    labels = np.zeros(norms.shape)
    labels[y, x] = 1
    
    # Get the data labels and create the Clusters structure
    indices = np.nonzero(labels)
    labels = indices[1].reshape(-1, 1)
    labeled = np.concatenate((data, labels), axis=-1)
    labeled = labeled[labeled[:, -1].argsort()]  # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    clusters = {i: None for i in range(k)}
    for i in range(k):
        arr = labeled[labeled[:, -1] == i]
        clusters[i] = arr[:, :-1]

    return clusters


#@time_func
def _generate_means(data: Clusterable, k: int, ndim:int) -> np.ndarray:
    """Randomly selects initial means with uniform distribution

    Args:
        data: The data from which the means are selected
        k: How many means to select
        ndim: Dimensionality of means

    Returns:
        np.ndarray: Initial Cluster Centroids

    Raises:
        ValueError: If can't find unique set of means.
    
    """
    COUNT_OUT = 1000
    count = 0


    while count < COUNT_OUT:
        indices = np.random.choice(np.arange(len(data)), size=(k,), replace=False) 
        means = data[indices, :ndim]
        if len(np.unique(means, axis=0)) == len(means):
            return np.array(means)
        count += 1
    else:
        raise ValueError("Could not find unique set of initial means.\n")


#@time_func
def _new_centroids(clusters: Clusters, ndim: int) -> np.ndarray:
    """Returns a new set of centroids
    
    Args:
        clusters: the current grouped data
        ndim: Dimension of data we are clustering

    Returns:
        np.ndarray: New Centroids
    
    """
    # If this becomes a bottleneck, replace the `for` loop with a list comp.
    centroids = []
    for key in clusters:
        cluster = clusters[key]
        avg = np.average(cluster[:, :ndim], axis=0)
        centroids.append(avg)
    else:
        return np.array(centroids)


#@time_func
def _validate(
        data: Union[Clusterable, list[Clusterable], tuple[Clusterable]],
        k: int,*,
        initial_means: np.ndarray = None,
        ndim: int = None,
        tolerance: float = 0.5,
        max_iterations: int = 100,
) -> tuple[Clusterable, np.ndarray, int]:
    """Perform validation checks on cluster arguments
    
    Args:
        data: The input data
        k: Amount of clusters desired
        initial_means: The initial cluster centroids
        ndim: Dimension limit for clustering. If default, the length of a given
            data element is used (all data dimensions clustered).
        tolerance: Max tolerable distance a centroid can move before requiring
            another round of clustering
        max_iterations: Max number of iterations before terminating function
            execution.

    Returns:
        np.ndarray, np.ndarray, int: Validated Data, Initial Centroids, ndim
        
    Raises:
        ValueError: if an input argument is incorrect in value
        TypeError: if an input argument is of the wrong type.

    """
    # Check k
    if not isinstance(k, int): 
        raise TypeError("k must be an integer.")
    if k < 1 or k > len(data):
        raise ValueError("k must be positive and can't exceed number of data points.")
    
    # Check tolerance
    if tolerance < SMALLEST_THRESH:
        raise ValueError("Tolerance too small.")
    
    # Check max_iterations
    if not isinstance(max_iterations, int):
        raise TypeError("Max iterations must be an integer.")
    if max_iterations < 0:
        raise ValueError("Max iterations must be greater than 0.")

    # Check data
    if not len(data):
        raise ValueError("No data provided.")
    try:
        new_data = np.array(data)
    except ValueError:
        raise ValueError("Input data must be homogeneous. All elements must have the same shape.")

    # Check ndim
    if ndim is None:
        ndim = new_data.shape[0]
    elif not isinstance(ndim, int):
        raise TypeError("Dimension parameter must be an integer.")
    elif ndim < 0:
        raise ValueError("Dimension parameter must be positive.")
    elif ndim > new_data.shape[0]:
        raise ValueError("Dimension value cannot exceed data dimensionality.")
    else:
        pass

    # Check initial means
    if initial_means is None:
        temp_means = _generate_means(new_data, k=k, ndim=ndim)
    else:
        # Create homogeneous data; Each entry is a row vector.
        temp_means = np.array([arr[:ndim] for arr in initial_means])
        temp_data = new_data[:, :ndim]
        assert temp_means.shape[-1] == temp_data.shape[-1]
        # check if all elements in the means are in the data
        """
        for centroid in temp_means:
            # Test if the centroid is one of the data entries in temp_data.
            try:
                assert np.any(np.equal(centroid, temp_data).all(1))
            except AssertionError:
                raise ValueError(f"Initial means item not among provided data: {centroid}")
        """
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
    return new_data, temp_means, ndim

validate = _validate
__all__ = ["_assign_clusters", "_validate", "_new_centroids", "_generate_means"]
