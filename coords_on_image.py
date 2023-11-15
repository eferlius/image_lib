import matplotlib.pyplot as plt
import cv2
import numpy as np
from . import crop_image
from . import filter_image
from . import plots_image
from . import rescale_image
from . import utils

def from_coords_tuple_to_tl_br(coords_tuple, returnInt = True):
    '''
    From tuples of coords of the type [(x1,y1,z1), (x2,y2,z2),...]
    to 2 lists:
        - tl: [min[x1,x2,...],min[y1,y2,...],min[z1,z2,...]]
        - br: [max[x1,x2,...],max[y1,y2,...],max[z1,z2,...]]
    If returnInt is True, the returned values are integers    

    Parameters
    ----------
    coords_tuple : list of tuples, 
        each tuple contains a set of coordinate
    returnInt : bool, optional
        if converting the outputs to int, by default True

    Returns
    -------   
    tl : list containing two elements: [x, y]
        top left coordinates
    br : list containing two elements: [x, y]
        bottom right coordinates
    '''
    tl = []
    br = []
    for i in range(len(coords_tuple[0])):
        #all the i-th coord of each tuple
        tmp = [x[i] for x in coords_tuple]
        if returnInt:
            # get the lowest value and put in tl
            tl.append(int(np.floor(np.amin(tmp))))
            # get the greates value and put in br
            br.append(int(np.ceil(np.amax(tmp))))
        else:
            # get the lowest value and put in tl
            tl.append(np.amin(tmp))
            # get the greates value and put in br
            br.append(np.amax(tmp))
    return tl, br

def from_coords_tuple_to_list_coords(coords_tuple, returnInt = True):
    '''
    From tuples of coords of the type [(x1,y1,z1), (x2,y2,z2),...]
    to 2 lists:
        - tl: [min[x1,x2,...],min[y1,y2,...],min[z1,z2,...]]
        - br: [max[x1,x2,...],max[y1,y2,...],max[z1,z2,...]]
    If returnInt is True, the returned values are integers    

    Parameters
    ----------
    coords_tuple : list of tuples, 
        each tuple contains a set of coordinate
    returnInt : bool, optional
        if converting the outputs to int, by default True

    Returns
    -------
    coords_list : list of coordinates
        [[x0, y0], [x1, y1], [x2, y2], [x3, y3] ...]
    '''
    coords_list = []
    for i in range(len(coords_tuple)):
        if returnInt:
            coords_list.append([int(round(coords_tuple[i][j])) for j in range(len(coords_tuple[i]))])
        else:
            coords_list.append([coords_tuple[i][j] for j in range(len(coords_tuple[i]))])
    return coords_list


    