import cv2
import numpy as np

def resize_percent(img, scale_percent = 300, interpolation = cv2.INTER_AREA):
    '''
    resizes an image so that the dimension of h and w are scale_percent % of the original dimension

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to be resized
    scale_percent : int, optional
        each dimension will have scale_percent % of the original dimension, 
        by default 300
    interpolation : cv2 constant, optional
        interpolation method, 
        by default cv2.INTER_AREA

    Returns
    -------
    resized image
    '''
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # rescale image
    return cv2.resize(img, dim, interpolation = interpolation)

def resize_max_pixel(img, maxPixels = 1000, interpolation = cv2.INTER_AREA):
    '''
    resizes an image so that the larger dimension has a number of pixels equal to maxPixels

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to be resized
    maxPixels : int, optional
        number of pixels of the max dimension of the larger dimension, by default 1000
    interpolation : cv2 constant, optional
        interpolation method, 
        by default cv2.INTER_AREA

    Returns
    -------
    resized image
    '''
    return resize_percent(img, int(100*maxPixels/np.max(img.shape[0:2])), interpolation = interpolation)
