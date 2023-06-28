#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:38:22 2023

@author: Tj
"""

from collections.abc import Iterable
from typing import ClassVar
import copy
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


class KMeans:    
    """
        A class for implementing k-Means clustering. 
        Clusters data and has the ability to perform image segmentation.
    """

    # =================
    # Class Variables
    # =================
    _THRESH_MAX: ClassVar[int] = 1

    # =================
    # Instance Variables
    # =================
    data: list
    segments: int
    threshold: float
    maxIterations: int
    initial_means: Iterable
    
    # =================
    # Initialization
    # =================
    def __init__(self, data, segments=2, initial_means=None, threshold=0.5, 
                 maxIterations=100):
        self._data = data
        self._segments = segments
        self._threshold = threshold
        self._maxIterations = maxIterations
        self._initial_means = initial_means
        
        self._validateParams()
        
        # Plotting (2D and 3D cases)
        self._figure2D = plt.figure();
        self._axes2D = self._figure2D.add_subplot();
        self._figure3D = plt.figure();
        self._axes3D = self._figure3D.add_subplot(projection='3d');
        # plt.ion()

    # ============
    # Properties
    # ============
    '''Use these instead of explicit getters and setters'''
    @property
    def data(self):
        """Returns a copy of the object's data"""
        return copy.deepcopy(self._data)
        
    @property
    def segments(self):
        """How many segments into which the data is clustered."""
        return self._segments

    @segments.setter
    def segments(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        old_val = self._segments
        try:
            self._segments = value
            self._validateParams()
        except ValueError:
            self._segments = old_val
            raise

    @property
    def threshold(self):
        """Threshold for k-means clustering."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if 0 <= value <= KMeans._THRESH_MAX:
            self._threshold = value
        else:
            raise ValueError(f'Threshold must be between 0 and {KMeans._THRESH_MAX}')
    
    @property
    def maxIterations(self):
        """Max number of iterations for k-Means clustering"""
        return self._maxIterations

    @maxIterations.setter
    def maxIterations(self, value: int):
        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        elif value < 1:
            raise ValueError("Value must be at least 1.")
        else:
            self._maxIterations = value

    # ===============
    # Class Methods
    # ===============
    
    # ::Public methods::
    def cluster(self,
                display: bool = True) -> list:
        """
        The main event; performs the data clustering operation.

        Parameters
        ----------
        display : bool, optional
            Whether to live-plot the data or not. The default is True.

        Returns
        -------
        list
            [Clusters, Centroids, IterationCount]

        """
        # Initialize Variables
        data = self.data
        K_NUM = self.segments
        THRESH = self.threshold
        
        # Check (preclude inf. loop)
        # If the length of the data is less than the target segment number, 
        # getting the initial means will result in an infinite loop. 
        # The program would never be able to get 
        # the target number of unique points.
        if len(data) < K_NUM:
            raise ValueError("Number of segments exceeds data points."\
                             " Ensure data is in a 1-D iterable.\n"\
                            "Length Data: {}, Segments: {}".format(len(data), 
                                                                   K_NUM)
                             )
        # print(f'Data: {data}\nSegments:{K_NUM}')
        
        # Declare variables
        clusters, centroids = None, None
        means = None
        
        # ===================
        # Set initial means
        # ===================
        
        # Perform checks on user-passed means.
        if self._initial_means is not None:            
            means = self._initial_means
        else:
            # get k random points to serve as initial means
            print("Generating initial means...")
            means_found = False
            
            # Loop until the mean points are found. It shouldn't loop at 
            # all since repeat points are unlikely, but with randomization, 
            # there's always a chance, so I placed the check for uniqueness.
            while not means_found:
                means = np.array([data[np.random.randint(0, len(data))]
                         for _ in range(K_NUM)])
                # Sanity Check
                print(means)
                length = len(means)
                assert length == K_NUM
                # Ensure no duplicate point
                check = np.unique(means, axis=0)
                if len(check) == length:
                    means_found = True
        # print(f'Means Check:\n{means}')   
        # Begin loop; Currently, the program will loop until each calculated
        # cluster centroid is within THRESH distance from the mean found in the
        # previous iteration, or until the iteration limit is reached,
        # whichever happens first.
        # I may change this calculation to use data variance in the future 
        # if that's more "official."
        thresh_reached = False
        iterations = 0
        print("Cluster Iteration Count:")
        while not thresh_reached:
            # Assign clusters
            iterations += 1
            if iterations >= self._maxIterations:
                print("Max iterations reached. Returning output.")
                thresh_reached = True
            print(iterations)
 
            clusters = self._assignLabels(means, data)
            # print(f'Clusters: {clusters}')
            centroids = self._findCentroid(clusters)
            # print(f'Centroids: {centroids}')
            # Live plot the data
            if display:
                self._display([clusters, centroids, iterations])
                
            # Compare centroids to previous means.
            for i in range(len(centroids)):
                distance = self._calcDistance(centroids[i], means[i])
                if distance > THRESH:
                    # Assign new means
                    means = centroids
                    break
            else:
                thresh_reached = True
        if iterations < self._maxIterations:
            print("Successful cluster operation.\n")
        return [clusters, centroids, iterations]    
    
    @staticmethod
    def segment_img(image: np.ndarray, clusters: dict, centroids: list,
                    random_colors: bool = False) -> np.ndarray:
        """
        Perform image segmentation from k-Means clustering

        Parameters
        ----------
        image : np.ndarray (RGB)
        clusters : dict
        centroids : list
        random_colors : bool, optional
            Whether to use the centroids as colors, or generate random ones.
            The default is False.
        Returns
        -------
        seg_img : np.ndarray
            The segmented image. (RGB)

        """
        # Setup: Copy the image and get the colors
        print(f'Beginning Image Segmentation: {len(clusters)} segments.')
        seg_img = np.copy(image)
        if random_colors:
            colors = [(np.random.randint(0, 256),
                      np.random.randint(0, 256),
                      np.random.randint(0, 256)) for _ in range(len(clusters))]
        else:
            # Use centroid color
            colors = np.round(centroids, 0)
            
        # Inspiration (and much gratitude):
        # https://stackoverflow.com/questions/16094563/numpy-get-index-where-value-is-true
        # Use Numpy nonzero function to find the indices of all elements that
        # match a given pixel in a cluster. 
        # Use those indices to replace the pixel values
        # therein with the segmentation color.
        
        seg_img = seg_img.reshape(-1, 3)
        for cluster in clusters:
            # Remember that the keys in the dictionary range from 0 to k-1, so they also
            # double as indices.
            print(f'Cluster {cluster}')
            seg_color = colors[cluster]
            for pixel in clusters[cluster]:
                # Get indices where image is equal to pixel
                # print(f'Pixel: {type(pixel)}')
                indices = np.nonzero(np.all(np.equal(pixel, seg_img), axis=1))
                # print(indices[0])
                seg_img[indices[0]] = seg_color
        else:
            # Please include the 'else' statement to prevent a tremendous 
            # debugging headache. Watch the indentation.
            seg_img = seg_img.reshape(image.shape)
        return seg_img

    # ::Private methods::
    def _validateParams(self):
        # access and validate the data
        data = self._data
        K_NUM = self._segments
        THRESH = self._threshold
        MAX_ITERATIONS = self._maxIterations
        initial_means = self._initial_means
        accepted_types = [list, tuple, np.ndarray]
        
        if np.array(data).ndim != 2:
            raise ValueError("Data *must* be w/in a 1-D container.\nEx. [(0, 0), (2,3)]")
        if K_NUM < 1:
            raise ValueError("Number of segments must be at least one.")
        elif K_NUM > len(data):
            raise ValueError("Number of segments cannot exceed number of data points.")
        if THRESH < 0 or THRESH > KMeans._THRESH_MAX:
            raise ValueError("Cannot have a negative threshold value.")
        if MAX_ITERATIONS <= 0:
            raise ValueError("Must have at least one iteration.")
            
        if initial_means is not None:
            if type(initial_means) not in accepted_types:
                raise TypeError(f'Means container must be one of the following:\n{accepted_types}')
            # Check element types and values.
            if not all([type(arr) in accepted_types for arr in initial_means]):
                raise TypeError(f'Elements must be one of the following types:\n{accepted_types}')
            if not all(any(np.equal(data, element).all(1)) 
                       for element in initial_means):
                raise ValueError("Provided means must be among the data.")
            # Remove duplicates and check remaining length
            # Turns out np.unique changes the order of the data.
            # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
            # print(f'Before: {initial_means}')
            initial_means, ndx = np.unique(initial_means, axis=1, 
                                           return_index=True)
            # print(ndx)
            initial_means = initial_means[:, ndx]
            # print(f'After {initial_means}')
            # Check length
            try:
                length = len(initial_means)
                assert length == K_NUM
            except AssertionError:
                print("\nNumber of mean points must == number of segments.\n")
                raise
            self._initial_means = initial_means
        print("KMeans: All parameters valid.")
        return True

    def _assignLabels(self, means: list, data: list):
        """
        Separate the data into clusters.

        Parameters
        ----------
        means : list
            The current list of cluster means. 
            Randomly chosen for first iteration.
        data : list
            The data to be organized.

        Returns
        -------
        clusters : dictionary

        """
        # Organizes the data into clusters based on which mean 
        # is closest to a given point.
        K_NUM = self._segments
        clusters = {k: [] for k in range(K_NUM)}
        index = None

        for point in data:
            # Initialize ridiculously high number to begin comparisons.
            old_dist = 1E1000 
            for i in range(len(means)):
                new_dist = self._calcDistance(point, means[i])
                
                if new_dist < old_dist:
                    # Track index of the closest mean point
                    (old_dist, index) = (new_dist, i)
            else:
                # Add point to label bin
                clusters[index].append(point)
                
        return clusters
        
    @staticmethod
    def _calcDistance(point1: Iterable, point2: Iterable = None):
        # check input
        if not isinstance(point1, Iterable):
            point1 = [point1]
        if point2 is None:
            point2 = np.zeros(len(point1))
        else:
            if not isinstance(point2, Iterable):
                point2 = [point2]
                
        # Cast as numpy arrays to prevent overflow
        point1 = np.array(point1, dtype=np.float64)
        point2 = np.array(point2, dtype=np.float64)
        
        try:
            assert len(point1) == len(point2)
        except AssertionError:
            print('\nBoth points must have same dimension.')
            raise
            
        # Perform Calculation  
        sqr_dist = (point1-point2)**2
        sqr_dist = sqr_dist.sum()
        dist = np.sqrt(sqr_dist)
        
        return dist
        
    @staticmethod
    def _findCentroid(clusters: Iterable):
        """
        Calculate the centroid for each cluster in the bin.
        Parameters
        ----------
        clusters : Iterable (dictionary)
            The clustered data.

        Returns
        -------
        centroids : list
            List of the centroids of each cluster

        """
        # Calculate the centroid for each cluster in the bin.
        # Takes in any iterable, but seeing as the data is labeled, it should be
        # a dictionary whose keys range from 0 to some n. 
        # However, I'm leaving it to work for more iterables in case I need to 
        # change the design in the future.
        
        centroids = []
        for i in range(len(clusters)):
            label_bin = np.array(clusters[i], dtype=np.float64)
            rows = label_bin.shape[0]
            # Calc centroid
            centroid = label_bin.sum(axis=0)/rows
            centroids.append(centroid)
        return centroids

    def _display(self, data: Iterable):
        ...
        # Assume we are passed the clusters, centroids, and iterations count
        # TO-DO: Figure out how to get the centroid to show in a 3D cluster.
        
        # Color list
        # Matplotlib Colors: 
        # https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_color_data.py
        """ 
        The WRAP_FACTOR causes the colors to be reused after exhaustion
        For example, if len(colors) == 10, but K_NUM == 11, 
        The 11th cluster will use the first value in colors because 
        10 % 10 == 0 (Remember 0-based indexing).
        """
        
        colors = [color for color in list(mcolors.TABLEAU_COLORS.values())]
        WRAP_FACTOR = len(colors)
        
        # Get data
        clusters, centroids, iterations = data
        # print(clusters[0][0])
        dimensions = len(clusters[0][0])
        # print(f'Data Dimensions: {dimensions}')
        labels = ['Cluster {}'.format(i) for i in range(len(clusters))]
        
        # 2D case
        if dimensions == 2:
            ax = self._axes2D
            ax.clear()
            ax.set(xlabel='x', ylabel='y',
                   title='k-Means Iteration {}\nk = {}'.format(iterations,
                                                               self._segments))
            for i in range(len(clusters)):
                x, y = zip(*clusters[i])
                cenX, cenY = centroids[i]
                ax.scatter(x, y, s=10, color=colors[i % WRAP_FACTOR],
                           label=labels[i])
                ax.scatter(cenX, cenY, color=colors[i % WRAP_FACTOR],
                           marker='o', s=50, zorder=3, edgecolor='k')
            ax.legend()
            # ax.grid(visible=True, axis='both')
            plt.pause(0.05)
            
        # 3D case
        elif dimensions == 3:
            
            ax = self._axes3D
            ax.clear()
            
            # Plot setup
            ax.set(xlabel='R', ylabel='G', zlabel='B',
                   title='k-Means Iteration {}\nk = {}'.format(iterations,
                                                               self._segments))
            # Get R, G, B points and plot them for each cluster
            # Matplotlib automatically switches color for each call to scatter.
            for i in range(len(clusters)):
                R, G, B = zip(*clusters[i])
                # cenR, cenG, cenB = centroids[i]
                ax.scatter(R, G, B, s=10, color=colors[i % WRAP_FACTOR])
                # ax.scatter(cenR, cenG, cenB ,marker='*', s=(72*2), zorder=5)
            plt.pause(0.05)
        # plt.show()
        else:
            # print("Data is neither 2D nor 3D. Returning.")
            return
    

class Image:
    """
        A class to make typical OpenCV operations simpler for myself.
    """

    # =================
    # Class Variables
    # =================
    
    # Incrementing this variable allows 
    # the class to display multiple image windows.
    _img_label: ClassVar[int] = 0

    # ====================
    # Instance Variables
    # ====================
    in_path: str

    # =================
    # Initialization
    # =================
    def __init__(self, in_path):
        
        self._img_path = self._format_path(in_path)
        
        # check if image exists on system
        if not os.path.isfile(self._img_path):
            raise ValueError("ERROR: Image file does not exist.")
            
        # set output path    
        self._output_dir, self._name = os.path.split(self._img_path)
        self._output_dir += '/'
        
        # Load image
        self._img_backup = cv.imread(self._img_path)
        assert self._img_backup is not None

    # ============
    # Properties
    # ============
    
    @property
    def output_dir(self):
        """
        The output directory of the image. Determines where operations are saved.
        """
        return self._output_dir
    @output_dir.setter
    def output_dir(self, path: str):
        if not path:
            raise ValueError('No path name provided.\nCurrent output directory:\n"{}"'.
                  format(self.output_dir))
        
        new_path = self._format_path(path)
        if not os.path.isdir(new_path):
            raise ValueError('\nCannot set output directory; it does not exist.')
                   
        else:
            # Formatting choice; makes image saving easier for users.
            if new_path[-1] != '/':
                new_path += '/'
            self._output_dir = new_path
            return

    @property
    def name(self):
        """
        Returns
        -------
        image_name: str
        """
        return self._name

    @property
    def img_backup(self) -> np.ndarray:
        """
        An untouched version of the passed image.

        Returns
        -------
        numpy.ndarray of image

        """
        return np.copy(self._img_backup)

    @property
    def size(self) -> tuple:
        """
        Returns
        -------
        tuple
           (height, width) in px.

        """
        height, width, _ = self._img_backup.shape
        return height, width
        
    # ===============
    # Class Methods
    # ===============
    
    # ::Public methods::
    
    @staticmethod
    def view_img(image: np.ndarray):
        """
        Displays an image passed to the class.

        Parameters
        ----------
        image : np.ndarray

        """
        if image is not None and isinstance(image, np.ndarray):
            cv.imshow("Image Class Display {}".format(Image._img_label), image) 
            cv.waitKey(1)
            Image._img_label += 1
        else:
            raise ValueError("No image passed.")

    def display(self):
        """Displays the original image."""
        self._display(self._img_backup)

    def save(self, image: np.ndarray,
             name: str = None) -> bool:
        """
        Description
        -----------
            Save an image to the output directory
        Example
        -------
            img._save(img.img_backup, 'test.png')
        """
        
        # Name Check
        if not name:
            raise ValueError("Could not save image. No name given.")
        
        # Set return boolean
        is_success = False
        
        # try the save operation
        saving_path = self._output_dir + name
        try:
            print("Saving file: '{}'\n...".format(saving_path))
            cv.imwrite(saving_path, image)
        except Exception as e:
            print("\nFailed to save image. Check the path or extension.\n",
                  e, sep='\n')
            return is_success
        else:
            # write was successful
            print("Save successful.\n")
            is_success = True
            return is_success

    # Image Operations
    def cvt_color(self, conversion: str,
                  get_data: bool = False,
                  display: bool = True) -> np.ndarray:
        """
        Convert the images color space.

        Parameters
        ----------
        conversion : str
            The space to convert to (ex. HSV). 
        get_data : bool, optional
            Whether to return the converted image as a numpy array. 
            The default is False.
        display : bool, optional
            Display the converted image. The default is True.

        Raises
        ------
        ValueError

        Returns
        -------
        converted_img : np.ndarray
            The converted image.
        """
        if not display and not get_data:
            return np.array([])
        # Check conversion validity
        color_codes = {'gray': cv.COLOR_BGR2GRAY,
                      'HSV': cv.COLOR_BGR2HSV,
                      'RGB': cv.COLOR_BGR2RGB}
        valid_entries = [*color_codes.keys()]
        
        try:
            if conversion in valid_entries:
                pass
            else:
                raise ValueError('Incorrect Conversion Type.\nValid options:\n{}'.format(valid_entries))
        except (TypeError, AttributeError):
            print("Incorrect Entry. Please use a string.\nValid Entries:")
            print(valid_entries)
            raise
        
        # Implement conversion
        work_img = np.copy(self._img_backup)
        converted_img = cv.cvtColor(work_img, color_codes[conversion])
        
        # Boolean Actions
        if display:
            self._display(converted_img)
        if get_data:
            return converted_img

    def color_rebalance(self, 
                        R_const: float, 
                        G_const: float,
                        B_const: float, 
                        display: bool = True, 
                        get_data=False) -> np.ndarray:
        """
        Re-balances the color channels of the image based on provided factors.        
        
        Parameters
        ----------
        R_const : float
            Red channel factor.
        G_const : float
            Green channel factor.
        B_const : float
            Blue channel factor.
        display : bool, optional
            Whether to show the image or not. The default is False.
        get_data :  bool, optional
            Return the modified image as an array or not.
        Raises
        ------
        ValueError
            Only insert non-negative numbers.

        Returns
        -------
        new_img : np.ndarray
            The rebalanced image.
            
        """
        if not display and not get_data:
            return np.array([])
        
        # Check inputs
        try:
            if R_const < 0 or G_const < 0 or B_const < 0:
                raise ValueError("Please insert non-negative numbers.")
        except TypeError:
            print("\nColor Re-balance Error: Please insert numbers.\n")
            return np.array([])

        constants = [B_const, G_const, R_const]
        # apply to image
        work_img = np.copy(self._img_backup)
        channels = list(cv.split(work_img))

        for i in range(len(constants)):
            channels[i] = (channels[i]*constants[i]).astype(np.uint8)
            
        # Channel validation
        # print([channels[i].dtype for i in range(len(channels))])
        
        new_img = cv.merge(channels)
        new_img = np.round(new_img).astype(np.uint8)
        
        # Display
        if display:
            self._display(new_img)
        if get_data:
            return new_img

    def get_color_space(self, duplicate_pixels: bool = False):
        """
        Parameters
        ----------
        duplicate_pixels : bool, optional
            Determines whether to keep duplicate pixels or not.
            The default is False.

        Returns
        -------
        points: list
            The [(R), (G), (B)] points of each pixel .

        """
        # Data containers
        
        # Fill data containers
        work_img = self.cvt_color('RGB', get_data=True, display=False)
        color_points = work_img.reshape(-1, 3)
    
        if not duplicate_pixels:
            # removes duplicates for uniform plotting.
            unique_points = np.unique(color_points, axis=0)  # For 2D array
            return unique_points
        else:
            return color_points

    def view_color_space(self, uniform: bool = False):
        """
        View color space of the image.

        Parameters
        ----------
        uniform : bool, optional
            Whether to plot the feature space with equal weighting.
            Returns a sample of the feature space otherwise.
            The default is False.
        """
        # Data containers
        if uniform:
            # Remove duplicates from color space
            points = self.get_color_space()
            
        else:   
            points = self.get_color_space(duplicate_pixels=True)
        
        R, G, B = zip(*points)
        # Data manipulation
        # Get Unique Data
        
        # Plot Configuration
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        height, width, _ = self._img_backup.shape
        step = int(height * width / 2000)
        ticks = list(range(0, 250, 50))
        ticks.append(255)

        # Figure Axis settings
        ax.set(xlabel='R', ylabel='G', zlabel='B', 
               xlim=(0, 255), ylim=(0, 255), zlim=(0, 255),
               xticks=ticks, yticks=ticks, zticks=ticks)
        
        # Plot the data
        name = os.path.splitext(self.name)[0]
        if uniform:
            ax.set_title("{}'s Feature Space".format(name))
            ax.scatter(R, G, B, color='k')
        else:
            ax.set_title("{}'s Sampled Feature Space".format(name))
            ax.scatter(R[::step], G[::step], B[::step], color='k')
        # Show plot
        plt.show();

    def transform(self, *, 
                  translation_vals: np.ndarray = None, 
                  angle: float = 0,
                  get_data: bool = False, 
                  display: bool = True) -> np.ndarray:
        """
        Applies an affine transformation to the image using a combination
        of translation and rotation

        Parameters
        ----------

        translation_vals : np.ndarray, optional
            Translation Parameters [delta_x, delta_y].
            The default is [0,0].
        angle : float, optional
            Rotates about the image center in degrees.
            The default is 0.
        get_data : bool, optional
            Return the transformed image as a numpy array. 
            The default is False.
        display : bool, optional
            Display the transformed image. The default is True.
       

        Returns
        -------
        transformed_img : np.ndarray

        """
        # 
        if not display and not get_data:
            return np.ndarray([])

        if translation_vals is None:
            translation_vals = np.zeros(2)

        transformed_img = np.copy(self._img_backup)
        
        # Generate transformation matrix
        t_x, t_y = translation_vals
        
        T_mat = np.array([[1, 0, t_x],
                          [0, 1, t_y],
                          [0, 0, 1]])
        
        rows, cols, _ = transformed_img.shape
        R_mat = cv.getRotationMatrix2D(((cols-1)/2, (rows-1)/2), angle, 1)
        R_mat = np.append(R_mat, [[0, 0, 1]], axis=0)
        M_matrix = T_mat @ R_mat 
        M_matrix = M_matrix[0:2]

        # Apply transformation
        transformed_img = cv.warpAffine(transformed_img, M_matrix, 
                                        (cols, rows))
        
        if display:
            self._display(transformed_img)
        if get_data:
            return transformed_img

    def SIFT(self, *, display: bool = True,
             in_color: bool = True, get_data: bool = False) -> list:
        """
        Performs SIFT matching on the image for a 90-degree rotation.

        Parameters
        ----------
        display : bool, optional
            Display output or not. The default is True.
        in_color : bool, optional
            Display in color or grayed. The default is True.
        get_data : bool, optional
            Return data in the form [matches, matched_img]; 
            Default False.

        Returns
        -------
        List with SIFT matches and the resulting image as a numpy array; Convenient for saving.

        """
        
        if not display and not get_data:
            return []
        
        # Get worker images
        work_img = np.copy(self._img_backup)
        ex_transform = self.transform(angle=90,
                                      get_data=True,
                                      display=False)
        gray = self.cvt_color('gray', get_data=True, display=False)
        gray_transformed = cv.cvtColor(ex_transform, cv.COLOR_BGR2GRAY)
        
        # Begin SIFT
        # Source: https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
        sift = cv.SIFT.create()
        kp1_1, des1_1 = sift.detectAndCompute(gray, None)
        kp1_2, des1_2 = sift.detectAndCompute(gray_transformed, None)
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1_1, des1_2, k=2)
        
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # Draw matches
        SIFT_out = cv.drawMatchesKnn(
            gray, kp1_1,
            gray_transformed, kp1_2,
            good, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        SIFT_out_color = cv.drawMatchesKnn(
            work_img, kp1_1,
            ex_transform, kp1_2,
            good, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        print(f'SIFT matches: {len(good)}')
        
        # Display and/or return data
        if in_color:
            if display:
                self._display(SIFT_out_color)
            if get_data:
                return [good, SIFT_out_color]
        else:
            if display:
                self._display(SIFT_out)
            if get_data:
                return [good, SIFT_out]

    def ORB(self, *, display: bool = True, keepPercent: float = 0.5,
             in_color: bool = True,
             get_data: bool = False) -> list:
        """
        Performs ORB matching on the image for a 90-degree rotation.

        Parameters
        ----------
        display : bool, optional
            Display the output image or not.
            The default is True.
        keepPercent: float, optional
            Percentage of best matches to keep.
            Defaults to 0.5
        in_color : bool, optional
            Display in color or grayed. The default is True.
        get_data : bool, optional
            Return data in the form [matches, matched_img]; 
            Default False.

        Returns
        -------
        ORB image as an numpy array; Convenient for saving.

        """
        
        if not display and not get_data:
            return []
        
        # check input
        if keepPercent > 1:
            keepPercent = 1
        elif keepPercent < 0:
            print("Invalid input; Defaulting to 0.5")
            keepPercent = 0.5

        # Get worker images
        work_img = np.copy(self._img_backup)
        ex_transform = self.transform(angle=90,
                                      get_data=True,
                                      display=False)
        gray = self.cvt_color('gray', get_data=True, display=False)
        gray_transformed = cv.cvtColor(ex_transform, cv.COLOR_BGR2GRAY)
        
        # Apply ORB
        # Source: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        orb = cv.ORB_create()

        kp2_1, des2_1 = orb.detectAndCompute(gray, None)
        kp2_2, des2_2 = orb.detectAndCompute(gray_transformed, None)

        bf2 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        ORB_matches = bf2.match(des2_1, des2_2)

        # Sort match descriptors in the order of their distance.
        ORB_matches = sorted(ORB_matches, key=lambda x: x.distance)
        keep = int(len(ORB_matches) * keepPercent)
        best_matches = ORB_matches[:keep]
        
        # Draw matches.
        ORB_out = cv.drawMatches(
            gray, kp2_1,
            gray_transformed, kp2_2,
            best_matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        ORB_out_color = cv.drawMatches(
            work_img, kp2_1,
            ex_transform, kp2_2,
            best_matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        print(f'Total ORB matches: {len(ORB_matches)}')
        print(f'Best ORB matches: {len(best_matches)}')
        
        # Display and/or return data
        if in_color:
            if display:
                self._display(ORB_out_color)
            if get_data:
                return [best_matches, ORB_out_color]
            
        else:
            if display:
                self._display(ORB_out)
            if get_data:
                return [best_matches, ORB_out]

    # ::Private methods::
    @staticmethod
    def _format_path(path):
        """

        Parameters
        ----------
        path : str
            The path of the input image.

        Returns
        -------
        new_path : str
            The properly-formatted path string for OpenCV. Should be 
            platform-agnostic (Windows, Unix)
        """
        try:
            new_path = path.strip("'")
            new_path = new_path.strip('"')
            new_path = new_path.replace("\\", "/")  # For Windows pathing
            new_path = new_path.strip('//')  
        except AttributeError:
            print("\nWrong Type; Please insert a string.")
            raise
        return new_path

    def _display(self, image: np.ndarray):
        """
        Displays the given image. Defaults to the original image.
        This method is private to prevent users from being able to display 
        non-instance-origin-ed images.
        
        Meant for internal use (as in displaying output of a transformation)
        """
        # Default image to display
        if image is None:
            image = self.img_backup
            
        print("Press 'q' or 'ESC' to exit.")
        
        while True:
            cv.imshow("{}'s Output".format(os.path.splitext(self.name)[0]),
                      image)
            key = cv.waitKey(30)
            if key == 27 or key == ord('q'):
                break
        # Cleanup
        cv.destroyAllWindows()
        print("Closed.")
