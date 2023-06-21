#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:38:22 2023

@author: Tj
"""

from collections.abc import Iterable
#from typing import ClassVar
import copy
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class KMeans:    
    '''
        Description:
    '''

    #=================
    # Class Variables
    #=================
    '''Global class variables go here'''
    
    #=================
    # Instance Variables
    #=================
    '''Data that varies by specific instance goes here'''
    data:list
    segments:int
    threshold:float
    maxIterations:int 
    #=================
    # Initialization
    #=================
    '''Self-Explanatory: Set up the class'''
    def __init__(self, data, segments=2, threshold=0.5, maxIterations=100):
        # Look into data checks later
        # TO_DO:
            # Find a way to check if the segement count surpasses the
            # total number of data points.
        self._data = data
        self._segments = segments
        self._threshold = threshold
        self._maxIterations = maxIterations
        # Plotting
        self._figure = plt.figure();
        self._axis = self._figure.add_subplot(projection='3d');
        
        
    #============
    # Properties
    #============
    '''Use these instead of explicit getters and setters'''
    @property
    def data(self):
        '''Returns a copy of the object's data'''
        return copy.deepcopy(self._data)
    @property
    def segments(self):
        '''How many segments into which the data is clustered.'''
        return self._segments

    @segments.setter
    def segments(self, value:int):
        seg_val = value
        try:
            if value < 1:
                raise ValueError("Number of segments must be greater than 0.")
        except (TypeError, AttributeError):
            print("Number of segments must be an integer value.")
            raise
        else:
            self._segments = int(seg_val)
            
    @property
    def threshold(self):
        '''Threshold for k-means clustering.'''
        return self._threshold
    @threshold.setter
    def threshold(self, value:float):
        lowerLim, upperLim = 0, 1
        if value >= lowerLim and  value <= upperLim:
            self._threshold = value
        else:
            raise ValueError(f'Threshold must be between [{lowerLim}, {upperLim}]')
    
    #===============
    # Class Methods
    #===============
    
    # ::Public methods::
    def cluster(self, display:bool=True):
        # Initialize Variables
        data = self.data
        k_num = self.segments
        THRESH = self.threshold
        
        # Check (preclude inf. loop)
        if len(data) < k_num:
            raise ValueError("Number of segments exceeds data points."\
                             " Ensure data is in a 1-D iterable.\n"\
                            "Length Data: {}, Segments: {}".format(len(data), 
                                                                   k_num)
                            )
        #print(f'Data: {data}\nSegments:{k_num}')
        
        # get k random points to serve as initial means
        print("Setting initial means...")
        means_found = False
        
        while not means_found:
            means = [data[np.random.randint(0, len(data))]
                     for i in range(k_num)]
            # Sanity Check
            length = len(means)
            assert length == k_num
            # Ensure no duplicate point
            check = set(means)
            if len(check) == length:
                means_found = True
            
        # Get label bins
        thresh_reached = False
        iterations = 0
        print("Iteration Count:")
        while not thresh_reached:
            # Assign labels
            iterations += 1
            print(iterations)

            if iterations >= self._maxIterations:
                print("Max iterations reached. Returning output.")
                thresh_reached= True
                
            labels = self._assignLabels(means, data)
            
            # Find centroid of each bin.
            centroids = self._findCentroid(labels)
            
            # Live plot the data
            if display:
                self._display([labels, centroids, iterations])
            # Compare centroid to previous mean.
            for i in range(len(centroids)):
                distance = self._calcDistance(centroids[i], means[i])
                if distance > THRESH:
                    # Assign new means
                    means = centroids
                    break
            else:
                thresh_reached = True
        if iterations < self._maxIterations:
            print("Successful cluster operation. Returning.")
        return [labels, centroids, iterations]
    ''' Should I have a calculate means function???'''    
    
    
    
    # ::Private methods::
    def _assignLabels(self, means:list, data:list):
        # Get relevant data
        k_num = self._segments
        labels = {k:[] for k in range(k_num)}
        
        # Check if data will work with distance calculator function
        '''if not isinstance(data[0], Iterable):
            data = [[item] for item in data]
            means = [[mean] for mean in means]'''
        
        for point in data:
            old_dist = 1E1000 # Ridiculously high number to initialize comparisons.
            #print('\n', point, sep='\n')
            for i in range(len(means)):
                new_dist = self._calcDistance(point, means[i])
                #print(i, new_dist)
                if old_dist > new_dist:
                    # Track index of closest mean point
                    (old_dist, index) = (new_dist, i)
            else:
                # Add point to label bin
                labels[index].append(point)
           
        return labels
        
        
    def _dataValidate(self):
        ...
        
        
    def _calcDistance(self, point1:Iterable, point2:Iterable=None):
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
        sqr_dist = [(point1[coord] - point2[coord])**2 
                    for coord in range(len(point1))]
        sqr_dist = sum(sqr_dist)
        dist = np.sqrt(sqr_dist)
        #print(f'Point 1: {point1}')
        #print(f'Point 2: {point2}')
        #print(f'Calculated Distance: {np.sqrt(sqr_dist)}')
        return dist
        
    
    def _findCentroid(self, points:Iterable):
        ...
        # Assume data is a list of tuple points (x,y,z), (R,G,B), etc.
        dim = len(points)
        centroids = []
        for i in range(dim):
            label_bin = np.array(points[i])
            rows = label_bin.shape[0]
            # Calc centroid
            centroid = label_bin.sum(axis=0)/rows
            centroid = centroid.astype(np.float64)
            centroids.append(centroid)
        return centroids
        
    
    def _display(self, data:Iterable):
        ...
        # Assume we are passed the clusters, centroids
        clusters, _, iterations = data
        # Get R, G, B points
        # Plot the clusters
        ax = self._axis
        #fig = plt.figure();
        #ax = fig.add_subplot(projection='3d');
        # Refresh axis
        ax.clear()
        # Plot setup
        ax.set(xlabel='R', ylabel='G', zlabel='B',
               title='k-Means Iteration {}'.format(iterations))
        for i in range(len(clusters)):
            R, G, B = zip(*clusters[i])
            #cenR, cenG, cenB = (centroids[i].round())
            ax.scatter(R,G,B, s=10)
            #ax.scatter(cenR, cenG, cenB ,marker='*', s=(72*2), zorder=5)
        plt.pause(0.05)
        #plt.show()
    
    

class Image:
    '''
        A class to make typical OpenCV operations simpler for myself.
    '''
# =============================================================================
#         TO-DO:
#           1.) Implement Transformations
#           2.) Add Keypoint Features (SIFT, ORB)
# =============================================================================
    
    
    #=================
    # Class Variables
    #=================
    
    #====================
    # Instance Variables
    #====================
    in_path: str
    
    #=================
    # Initialization
    #=================
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
        
        
    #============
    # Properties
    #============

    
    @property
    def output_dir(self):
        '''
        The output directory of the image. Determines where operations are saved.
        '''
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
        '''
        Returns
        -------
        image_name: str
        '''
        return self._name


    @property
    def img_backup(self)-> np.ndarray: 
        '''
        An untouched version of the passed image.

        Returns
        -------
        numpy.ndarray of image

        '''
        return np.copy(self._img_backup)

    @property
    def size(self)->tuple:
        '''
        Returns
        -------
        tuple
           (height, width) in px.

        '''
        height, width, _ = self._img_backup.shape
        return (height, width)
        
    #===============
    # Class Methods
    #===============
    
    # ::Public methods::
    
    def display(self):
        '''Displays the original image.'''
        self._display(self._img_backup)
    
    
    def save(self, image: np.ndarray,
             name:str = None)->bool:
        '''
        Description
        -----------
            Save an image to the output directory
        Example
        -------
            img._save(img.img_backup, 'test.png')
        '''
        
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
    def cvt_color(self, conversion:str, 
                  get_data:bool= False,
                  display:bool = True)->np.ndarray:
        '''
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
        '''
        if not display and not get_data:
            return
        # Check conversion validity
        color_codes= {'gray':cv.COLOR_BGR2GRAY, 
                      'HSV':cv.COLOR_BGR2HSV, 
                      'RGB':cv.COLOR_BGR2RGB}
        valid_entries = [*color_codes.keys()]
        
        try:
            if (conversion in valid_entries):
                pass
            else:
                raise ValueError('Incorrect Conversion Type.\n'\
                                 'Valid options:\n{}'.format(valid_entries))
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
        
    
    
    def color_rebalance(self, R_const:float, G_const:float,
                        B_const:float, display:bool=True, get_data=False)->np.ndarray:
        '''
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
            
        '''
        if not display and not get_data:
            return
        
        # Check inputs
        try:
            if R_const < 0 or G_const < 0 or B_const < 0:
                raise ValueError("Please insert non-negative numbers.")
        except TypeError:
            print("\nColor Rebalance Error: Please insert numbers.\n")
            return
            
            
        constants = [B_const, G_const, R_const]
        # apply to image
        work_img = np.copy(self._img_backup)
        channels = list(cv.split(work_img))

        for i in range(len(constants)):
            channels[i] = (channels[i]*constants[i]).astype(np.uint8)
            
        # Channel validation
        #print([channels[i].dtype for i in range(len(channels))])
        
        new_img = cv.merge(channels)
        new_img = np.round(new_img).astype(np.uint8)
        
        # Display
        if display:
            self._display(new_img)
        if get_data:
            return new_img
    
    
    def get_color_space(self, duplicate_pixels:bool = False):
        '''
        Parameters
        ----------
        duplicate_pixels : bool, optional
            Determines whether to keep duplicate pixels or not.
            The default is False.

        Returns
        -------
        points: list
            The [(R), (G), (B)] points of each pixel .

        '''
        # Data containers
        color_points = []
        
        # Fill data containers
        work_img = self.cvt_color('RGB', get_data=True, display=False)
        for row in work_img:
            for col in row:
                r,g,b = col
                color_points.append((r,g,b))
        
        if not duplicate_pixels:
            # removes duplicates for uniform plotting.
            unique_points = [*set(color_points)] 
            print("Removing Duplicate Points.")
            return unique_points
        else:
            print("Returning all points")
            return color_points
    
    
    def view_color_space(self, uniform:bool=False):
        '''
        BETA:
            View sample of color space of the image.

        Parameters
        ----------
        uniform : bool, optional
            Whether to plot the feature space with equal weighting.
            Returns a sample of the feature space otherwise.
            The default is False.

        Returns
        -------
        None.

        '''
        # Data containers
        if uniform:
            # Remove duplicates from color space
            points = self.get_color_space()
            
        else:   
            points = self.get_color_space(duplicate_pixels=True)
        
        R, G, B = [], [], []
        for point in points:
            R.append(point[0])
            G.append(point[1])
            B.append(point[2])
        # Data manipulation
        # Get Unique Data
        
        # Plot Configuration
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        height, width, _ = self._img_backup.shape
        step = int(height*width/2000)
        ticks = list(range(0,250,50))
        ticks.append(255)

        # Figure Axis settings
        ax.set(xlabel='R', ylabel='G', zlabel='B', 
               xlim = (0,255), ylim = (0,255), zlim = (0,255),
               xticks = ticks, yticks =ticks, zticks = ticks)
        
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
        
    
        
    def transform(self,*,translation_vals:np.ndarray=[0,0], angle:float=0,
                  get_data:bool=False, display:bool=True):
        '''
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

        '''
        # 
        if not display and not get_data:
            return
        
        transformed_img = np.copy(self._img_backup)
        
        # Generate transformation matrix
        t_x,t_y = translation_vals
        
        T_mat = np.array([[1, 0, t_x],
                          [0, 1, t_y],
                          [0, 0, 1]])
        
        rows, cols, _ = transformed_img.shape
        R_mat = cv.getRotationMatrix2D(((cols-1)/2, (rows-1)/2), angle, 1)
        R_mat = np.append(R_mat, [[0,0,1]], axis=0)
        M_matrix = T_mat @ R_mat 
        M_matrix = M_matrix[0:2]

        # Apply transformation
        transformed_img = cv.warpAffine(transformed_img, M_matrix, 
                                        (cols, rows))
        
        if display:
            self._display(transformed_img)
        if get_data:
            return transformed_img
        
        
    def SIFT(self, *,display:bool=True, 
             in_color:bool=True, get_data:bool=False)->np.ndarray:
        '''
        Performs SIFT matching on the image for a 90 degree rotation.

        Parameters
        ----------
        display : bool, optional
            DESCRIPTION. The default is True.
        in_color : bool, optional
            Display in color or grayed. The default is True.
        get_data : bool, optional
            Return data in the form [matches, matched_img]; 
            Default False.

        Returns
        -------
        SIFT image as an numpy array; Convenient for saving.

        '''
        
        if not display and not get_data:
            return
        
        # Get worker images
        work_img = np.copy(self._img_backup)
        ex_transform = self.transform(angle=90,
                                      get_data=True,
                                      display=False)
        gray = self.cvt_color('gray', get_data=True, display=False)
        gray_transformed = cv.cvtColor(ex_transform, cv.COLOR_BGR2GRAY)
        
        # Begin SIFT
        sift = cv.SIFT.create()
        kp1_1, des1_1 = sift.detectAndCompute(gray, None)
        kp1_2, des1_2 = sift.detectAndCompute(gray_transformed, None)
        
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1_1, des1_2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
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
            work_img,kp1_1,
            ex_transform,kp1_2,
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
        

    def ORB(self, *,display:bool=True, keepPercent:float=0.5,
             in_color:bool=True,
             get_data:bool=False)->np.ndarray:
        '''
        Performs ORB matching on the image for a 90 degree rotation.

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

        '''
        
        if not display and not get_data:
            return
        
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
        orb = cv.ORB_create()

        kp2_1, des2_1 = orb.detectAndCompute(gray, None)
        kp2_2, des2_2 = orb.detectAndCompute(gray_transformed, None)

        bf2 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        ORB_matches = bf2.match(des2_1, des2_2)


        # Sort match descriptors in the order of their distance.
        ORB_matches = sorted(ORB_matches, key = lambda x:x.distance)
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
    def _format_path(self, path):
        '''

        Parameters
        ----------
        path : str
            The path of the input image.

        Returns
        -------
        new_path : str
            The properly-formatted path string for OpenCV. Should be 
            platform agnostic (Windows, Unix)
        '''
        try:
            new_path = path.strip("'")
            new_path = new_path.strip('"')
            new_path = new_path.replace("\\", "/") # For Windows pathing
            new_path = new_path.strip('//')  
        except AttributeError:
            print("\nWrong Type; Please insert a string.")
            raise
        return new_path
    
        
    def _display(self, image:np.ndarray):
        '''
        Displays the given image. Defaults to the original image.
        This method is private to prevent users from being able to display 
        non-instance-origined images.
        
        Meant for internal use (as in displaying output of a transformation)
        '''
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
        