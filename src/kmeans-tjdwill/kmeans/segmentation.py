#=============================================================================
#@author: tjdwill
#@date: 12 April 2024
#@title: Image segmentation
#@description:
#    Perform semantic segmentation on the provided image.
#=============================================================================
import itertools
#-
import numpy as np
#-
from kmeans import cluster


def _append_coords(img: np.ndarray) -> np.ndarray:
    """Append each pixel's coordinate to itself.
    
    Args:
        img: The image.
    
    Returns:
        np.ndarray: Array with appended indices.

    """
    height, width, *rest = img.shape
    indices = list(itertools.product(range(height), range(width)))
    indices = np.array(indices).reshape(height, width, -1)
    return np.concatenate((img, indices), axis=-1)


def segment_img(img: np.ndarray, groups: int, random_colors: bool = False) -> np.ndarray:
    """Segment the input RGB image by color groups.
    
    Args:
        img: The image to be segmented. Assumes RGB
        groups: How many groups the image is segmented into. Higher numbers -> more detail
        random_colors: Provide each group with a randomized RGB color instead of the average color.

    Returns:
        np.ndarray: Segmented Image
 
    """
    img_w_idxs = _append_coords(img)
    elem_dim = img_w_idxs.shape[-1]
    assert elem_dim == 5  #(R, G, B, y, x)
    color_groups, group_colors, *extra = cluster(img_w_idxs.reshape(-1, elem_dim), k=groups, ndim=3, tolerance=0.01)

    # The idea here is to use advanced indexing to change each group's color at once.
    '''
    Example:
    >> a = np.arange(9).reshape(3, 3)
    array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
    >> a[np.array([0, -1]), np.array([0, -1])] = 10
    >> a
    array([[10,  1,  2],
       [ 3,  4,  5],
       [ 6,  7, 10]])
    '''
    seg_img = np.zeros(shape=img.shape).astype(np.uint8)
    if random_colors:
        for key in color_groups:
            pixels_and_coords = color_groups[key]
            color = np.random.randint(0, 256, size=3).astype(np.uint8) 
            idxs = np.array([arr[-2:] for arr in pixels_and_coords])
            y, x = idxs[:, 0], idxs[:, 1]
            seg_img[y, x] = color
    else:
        for key in color_groups:
            pixels_and_coords = color_groups[key]
            color = group_colors[key].astype(np.uint8)
            idxs = np.array([arr[-2:] for arr in pixels_and_coords])
            y, x = idxs[:, 0], idxs[:, 1]
            seg_img[y, x] = color
    return seg_img