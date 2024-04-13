# K Means Clustering
\***Originally written as a class, the implementation has now been refactored to be function-based.**\*


A repository documenting the implementation of k-Means clustering in Python. Usage examples can be found in the `tests` directory.


The thing that makes this k-means clustering module different from others is that it allows the user to specify the number of dimensions to use for the clustering operation.

For example, given some data where each element is of form 
```python
# Each element would actually be a Numpy array, but the following uses lists for readability.
[
  [1, 2, 3, 4, 5],
  [4, 6, 7, 8, 2],
  ...
]
```
specifying `ndim=3` will result in only the first three elements of each data point being used for each operation.
This is useful for maintaining data association where it otherwise would be shuffled. An example of this is found in my implementation of image segmentation (`segmentation.py`) in this same project.
Other examples of use could be for maintaining data association in object detection elements. Given some 
```python
[xmin, ymin, xmax, ymax, conf, label]  # [bounding box, conf, label]
```
we may want to cluster the data solely on bounding box information while also maintaining the confidence intervals for each detection for further processing.


Example: k-Means implementation

2-D Case (1x speed)

[kmeans2D_animate.webm](https://github.com/tjdwill/KMeans_Clustering/assets/118497355/baf6e02a-4c28-4754-918e-60766a596911)

3-D Case (2.5x speed)

[kmeans3D_animate.webm](https://github.com/tjdwill/KMeans_Clustering/assets/118497355/22394f93-a2f3-499f-a54c-286723dd0a70)

## Image Segmentation
Perform image segmentation based on color groups specified by the user.

Two options: 
1. Averaged Colors
<img src="https://github.com/tjdwill/KMeans_Clustering/blob/main/tests/output/seg_groups4.jpg" />  
2. Random Colors
<img src="https://github.com/tjdwill/KMeans_Clustering/blob/main/tests/output/seg_rand_groups4.jpg" />

## Developed With
* Python (3.12.1)
* Numpy (1.26.2) 
* Matplotlib (3.8.4)

However, I don't use any features that are specific to Python 3.12.
