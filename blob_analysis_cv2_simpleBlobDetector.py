import pandas as pd
import cv2
import numpy as np
import skimage
from . import plots_image
from . import utils

"""
date: 2024-06-19 11:38:55
note: use contour
contours, hierarcy = cv2.findContours(frame[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
"""
             
class detected_blobs_info:
    def __init__(self, img, detector):
        '''
        Class of blob detection

        Parameters
        ----------
        img : matrix width*height*1
            image where the blobs_info will be computed
        detector : detector object
            example:
            # Setup the blob detector with desired parameters
            params = cv2.SimpleBlobDetector_Params()
        
            # Enable Gaussian filtering and set the standard deviation
            params.filterByColor = False
            params.filterByCircularity = False
            params.filterByInertia = False
            params.filterByConvexity = False
            params.filterByArea = True
            params.minArea = 100  # Adjust the minimum area threshold
            params.maxArea = 2000  # Adjust the maximum area threshold

            # Create the blob detector with the specified parameters
            detector = cv2.SimpleBlobDetector_create(params)
        '''
    
        # Detect blobs in the image
        keypoints = detector.detect(img)
        
        # Get blob information
        self.blobs_info = [[int(keypoint.pt[0]), int(keypoint.pt[1]), keypoint.size] for keypoint in keypoints]

        self.dictionary = self.create_dictionary()

    def create_dictionary(self):
        dictionary = {}
        dictionary['xc'] = [i[0] for i in self.blobs_info]
        dictionary['yc'] = [i[1] for i in self.blobs_info]
        dictionary['size'] = [i[2] for i in self.blobs_info]
        return dictionary      

def compute_detected_blobs_info_to_df(img, detector, show = False, return_also_image = False):
    '''
    Given a an image, compute the blobs
    Returns a pandas dataframe containing index, xc, yc and size of each blob

    Parameters
    ----------
    img : matrix width*height*1
        image where the blobs_info will be computed
    detector : detector object
        example:
        # Setup the blob detector with desired parameters
        params = cv2.SimpleBlobDetector_Params()
    
        # Enable Gaussian filtering and set the standard deviation
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByArea = True
        params.minArea = 100  # Adjust the minimum area threshold
        params.maxArea = 2000  # Adjust the maximum area threshold

        # Create the blob detector with the specified parameters
        detector = cv2.SimpleBlobDetector_create(params)

    Returns
    -------
    Pandas dataframe containing index, xc, yc and size of each blob
    '''
    img_1ch = utils.convert_to_grayscale(img)

    output = detected_blobs_info(img_1ch, detector)
    df = pd.DataFrame(output.dictionary, columns = list(output.dictionary.keys()))
    df.insert(0, 'blob num', np.arange(0,df.shape[0]))
    if show or return_also_image:
        img_plot = draw_blobs_on_img(img, output.blobs_info)
        if show: 
            plots_image.plot_image(img_plot, 'blobs  with simpleBlobDetector method')
        if return_also_image:
            return df, img_plot
    return df

def draw_blobs_on_img(img, blobs_info, colour = None, thickness = 2):
    if not colour:
        colour = [0, 0, 255]
    
    img_copy = img.copy()
    i = -1
    for blob in blobs_info:
        i+=1
        x, y, r = blob
        cv2.circle(img_copy, (int(x), int(y)), int(r), colour, thickness)
        cv2.putText(img_copy, str(i), (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, thickness, cv2.LINE_AA)
    return img_copy






