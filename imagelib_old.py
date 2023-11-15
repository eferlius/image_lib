# -*- coding: utf-8 -*-
"""
Library for fast operations on images
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from . import utils
from . import plots

def getTLBRprojection(img, discValue = 0, showPlot = False):
    '''
    Returns tl and br as indexes of first and last values different from 
    discValue (discarded value) in projection of the image
    

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    discValue : int or float, depending on image, optional
        discarded value. The default is 0.
    showPlot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    tl : TYPE
        DESCRIPTION.
    br : TYPE
        DESCRIPTION.

    '''
    hProj, vProj = projection(img, showPlot)

    # give widest dimensions
    tl = [0, 0]
    br = img.shape[0:2]
    br = [br[1], br[0]]

    try:
        tl[0] = np.argwhere(vProj!=discValue)[0][0]
    except:
        pass
    try:
        tl[1] = np.argwhere(hProj!=discValue)[0][0]
    except:
        pass
    try:
        br[0] = np.argwhere(vProj!=discValue)[-1][0]
    except:
        pass
    try:
        br[1] = np.argwhere(hProj!=discValue)[-1][0]
    except:
        pass

    return tl, br

def findStartStopValues(array, discValue = 0, maxIntervals = 100):
    '''
    Returns two lists: start and stop.
    - start contains all the indexes where the array passes from discardedValue 
    to another value
    - stop contains all the indexes where the array passes from another value
    to discardedValue

    If they're longer than maxInvtervals, they're reduced deleting both stop and 
    start of the closest stop to its consecutive start

    _extended_summary_

    Parameters
    ----------
    array : _type_
        _description_
    discValue : int, optional
        _description_, by default 0
    maxIntervals : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    '''
    start = []
    stop = []

    for i in np.arange(1, len(array)-1, 1):
        if array[i-1] == discValue and array[i] != discValue:
            start.append(i)
        if array[i] != discValue and array[i+1] == discValue:
            stop.append(i)

    # in case the array starts or ends without discarded values,
    # it's necessary to add them
    if len(start) == 0 or len(stop) == 0:
        if len(start) == 0:
            start.insert(0, 0)
        if len(stop) == 0:
            stop.append(len(array))
    else:
        if start[0] > stop[0]:
            start.insert(0, 0)
        if start[-1] > stop[-1]:
            stop.append(len(array))
        # might give problems
        # todo check this
    start, stop = reduceStartStopMinDist(start, stop, maxIntervals)

    return start, stop

def reduceStartStopMinDist(start, stop, maxIntervals = 100):
    assert len(start) == len(stop), f"start and stop should be of the same length, got {len(start)} and {len(stop)}"
    start = np.array(start)
    stop = np.array(stop)
    while len(start) > maxIntervals:
        closestIndexBetweenStartStop = np.argmin(start[1:]-stop[0:-1])
        start = np.delete(start, closestIndexBetweenStartStop+1)
        stop = np.delete(stop, closestIndexBetweenStartStop)

    return start, stop

# todo finish this
def getTLBRprojectionInside(img, discValue = 0, showPlot = False, maxIntH = 1, maxIntV = 3):
    '''
    Given an image, exectues the projection and return a list of tl and br coord
    according to the discValue

    _extended_summary_

    Parameters
    ----------
    img : _type_
        _description_
    discValue : int, optional
        _description_, by default 0
    showPlot : bool, optional
        _description_, by default False
    maxIntH : int, optional
        _description_, by default 1
    maxIntV : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    '''
    
    assert len(img.shape)==2, f"img should be 2 dimensional, got {img.shape}"
    hProj, vProj = projection(img, showPlot)

    starth, stoph = findStartStopValues(hProj, discValue, maxIntervals = maxIntH)
    startv, stopv = findStartStopValues(vProj, discValue, maxIntervals = maxIntV)
    tl_list = []
    br_list = []
    for tly, bry in zip(starth, stoph):
        for tlx, brx in zip(startv, stopv):
            tl_list.append([tlx, tly])
            br_list.append([brx, bry])

    return tl_list, br_list

def correctBorderLoop(img, startPointFlag, trueValue, replaceValue, showPlot = False):
    '''
    Starting from startPointFlag (top left, bottom left, bottom right or top right), 
    iteratively searches for pixel with trueValue and substitute them with replaceValue

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    startPointFlag : TYPE
        DESCRIPTION.
    trueValue : TYPE
        DESCRIPTION.
    replaceValue : TYPE
        DESCRIPTION.
    showPlot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    '''
    assert startPointFlag in ['tl', 'bl', 'br', 'tr'],f"startPoingFlag can only\
be ['tl', 'bl', 'br', 'tr'], got {startPointFlag}"

    trueValue = utils.make_list(trueValue)
    replaceValue = utils.make_list(replaceValue)

    origImg = img.copy()
    try:
        h, w = img.shape
    except:
        h, w, d = img.shape
    # starting from the top
    if startPointFlag[0] == 't':
        y0 = 0
    # starting from the bottom
    elif startPointFlag[0] == 'b':
        y0 = h-1
    #starting from left
    if startPointFlag[1] == 'l':
        x0 = 0
    # starting from right
    elif startPointFlag[1] == 'r':
        x0 = w-1

    if (img[y0,x0]==trueValue).all():
        allPoints = [[y0,x0]]
        startPoints = allPoints.copy()
        while True:
            newPoints = []
            startPoints = utils.remove_duplicates_from_list_of_list(startPoints)
            for y, x in startPoints:
                for coord in [[y,x+1],[y,x-1],[y+1,x],[y-1,x]]:
                    try:
                        if (img[coord[0], coord[1]]==trueValue).all() and coord not in allPoints:
                            allPoints.append(coord)
                            newPoints.append(coord)
                    except:
                        pass
            if newPoints != []: # at least one new point was found
                startPoints = newPoints.copy()
            else:
                break

        for y,x in allPoints:
            img[y,x]=np.array(replaceValue)

    if showPlot:
        plots.pltsImg([origImg, img], listTitles = ['original', 'after border correction'],\
                            mainTitle = 'application of border correction')

    return img

def correctBorderAllCorners(img, trueValue, replaceValue, showPlot = False):
    origImg = img.copy()
    trueValue = utils.make_list(trueValue)
    try:
        h, w = img.shape
    except:
        h, w, d = img.shape
        
    borderCorrectedFlag = [] # to tell in which corners a border detection was needed
    for startPointFlag, coord in zip(['tl', 'bl', 'br', 'tr'],[[0,0],[h-1,0],[h-1,w-1],[0,w-1]]):
        if (img[coord[0],coord[1]] == trueValue).all():
            borderCorrectedFlag.append(1)
            img = correctBorderLoop(img, startPointFlag, trueValue, replaceValue, showPlot)
        else:
            borderCorrectedFlag.append(0)
    return img, borderCorrectedFlag
#%% 
"""
date: 2022-12-22 14:42:55
note: when putting order in utils
"""
# def depImgToThreeCol(image):
#     '''
#     From a dep image, containing only 1 value per pixel: 

#         |----------------------------...------> x
#         |0.0       1.0       2.0     ...  img_w.0    
#         |0.1       1.1       2.1     ...  img_w.1 
#         |0.2       1.2       2.2     ...  img_w.2 
#         |0.3       1.3       2.3     ...  img_w.3 
#         ...        ...       ...     ...  ...
#         |0.img_h   1.img_h   2.img_h ...  img_w.img_h
#         v y

#     returns an 2D array with 3 columns (pointCloud):
#         x         y       dep
#         0         0       0.0
#         1         0       1.0
#         2         0       2.0
#         ...       ...     ...
#         img_w     0       img_w.0
#         --------------------------- first row of the image
#         0         1       0.1
#         1         1       1.1
#         2         1       2.1
#         ...       ...     ...
#         img_w     1       img_w.1
#         --------------------------- second row of the image
#         ...
#         ...
#         0         img_h   0.img_h
#         1         img_h   1.img_h
#         2         img_h   2.img_h
#         ...       ...     ...
#         img_w     img_h   img_w.img_h
#         --------------------------- last row of the image

#     Parameters
#     ----------
#     image : matrix
#         contains z values.

#     Returns
#     -------
#     data : array
#         contains x y z values.

#     '''
#     image_h, image_w = image.shape

#     hline = np.arange(0, image_w, 1)
#     xmask = np.repeat([hline], image_h, axis = 0)

#     vline = np.expand_dims(np.arange(0, image_h, 1), axis = 1)
#     ymask = np.repeat(vline, image_w, axis = 1)
    
#     # create a matrix x y z
#     data = np.zeros([image_h*image_w,3])
#     data[:,0] = xmask.flatten()
#     data[:,1] = ymask.flatten()
#     data[:,2] = image.flatten()

#     return data

# def threeColToDepImg(data, x_col_index = 0, y_col_index = 1, z_col_index = 2):
#     '''
#     From an 2D array with 3 columns:
#         x         y       dep
#         0         0       0.0
#         1         0       1.0
#         2         0       2.0
#         ...       ...     ...
#         img_w     0       img_w.0
#         --------------------------- first row of the image
#         0         1       0.1
#         1         1       1.1
#         2         1       2.1
#         ...       ...     ...
#         img_w     1       img_w.1
#         --------------------------- second row of the image
#         ...
#         ...
#         0         img_h   0.img_h
#         1         img_h   1.img_h
#         2         img_h   2.img_h
#         ...       ...     ...
#         img_w     img_h   img_w.img_h
#         --------------------------- last row of the image

#     returns a dep image, containing only 1 value per pixel: 

#         |----------------------------...------> x
#         |0.0       1.0       2.0     ...  img_w.0    
#         |0.1       1.1       2.1     ...  img_w.1 
#         |0.2       1.2       2.2     ...  img_w.2 
#         |0.3       1.3       2.3     ...  img_w.3 
#         ...        ...       ...     ...  ...
#         |0.img_h   1.img_h   2.img_h ...  img_w.img_h
#         v y

#      Parameters
#      ----------
#      data : array
#          contains x y z values.
#     x_col_index : int, optional
#         column of the x values. The default is 0.
#     y_col_index : int, optional
#         column of the y values. The default is 1.
#     z_col_index : int, optional
#         column of the z values. The default is 2.

#     Returns
#     -------
#     image : matrix
#         contains z values.

#     '''
#     # # removing nan values
#     # data = data[~np.isnan(data).any(axis=1), :]

#     image = np.reshape(data[:,z_col_index], [int(round(data[-1,y_col_index])+1),int(round(data[-1,x_col_index])+1)],'C')
#     return image
