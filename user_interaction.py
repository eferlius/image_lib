import matplotlib.pyplot as plt
import cv2
import numpy as np
from . import coords_on_image
from . import crop_image
from . import filter_image
from . import info_image
from . import plots_image
from . import rescale_image
from . import utils

def get_coords_user_tl_br(img, nPoints = -1, title = '', max_pixel_rescale = 800, returnInt = True):
    '''
    User can tap nPoints on the image, tl and br coordinates will be obtained from them.
    *advice*: use 3 points: one for tl, one for br and the third one in the middle to confirm

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        _description_
    nPoints : int, optional
        _description_, by default -1
    title : str, optional
        _description_, by default ''

    Returns
    -------   
    tl : list of [x, y]
        top left coordinates
    br : list of [x, y]
        bottom right coordinates
    '''
    orig_img = img.copy()
    h, w = img.shape[:2]
    tl = [0, 0]
    br = [w, h]

    while True:
        fig, ax = plots_image.plot_image(orig_img, title = title+'select the {} points'.format(nPoints))
        coords_tuple = plt.ginput(n=nPoints, timeout=-1, show_clicks=True)
        tl, br = coords_on_image.from_coords_tuple_to_tl_br(coords_tuple, returnInt = False)
        tl_int = list(map(int, tl))
        br_int = list(map(int, br))
        img = crop_image.crop_img_tl_br(orig_img, tl_int, br_int)

        imgName = 'Press Enter to confirm'
        cv2.imshow(imgName, rescale_image.resize_max_pixel(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_pixel_rescale))
        key = cv2.waitKey(0)
        if key == ord('\r'): # enter key
            plt.close(fig)
            cv2.destroyWindow(imgName)
            break
        else:
            plt.close(fig)
            cv2.destroyWindow(imgName)
            continue
    if returnInt:
        tl = tl_int
        br = br_int
    else: 
        pass
    return tl, br

def get_coords_user_contour(img, nPoints = -1, select = 'inside', exclusion_colour = (0, 0, 0), 
                            title = '', max_pixel_rescale = 800, returnInt = True):
    orig_img = img.copy()
    h, w = img.shape[:2]
    tl = [0, 0]
    br = [w, h]

    while True:
        fig, ax = plots_image.plot_image(orig_img, title = title+'select the {} points'.format(nPoints))
        coords_tuple = plt.ginput(n=nPoints, timeout=-1, show_clicks=True)
        tl, br = coords_on_image.from_coords_tuple_to_tl_br(coords_tuple, returnInt = False)
        coords_list = coords_on_image.from_coords_tuple_to_list_coords(coords_tuple, returnInt = False)
        tl_int = list(map(int, tl))
        br_int = list(map(int, br))
        coords_list_int = [[int(num) for num in sub] for sub in coords_list]

        img = orig_img.copy()
        img = filter_image.get_selected_pixels_from_contour(img, np.array(coords_list_int), select = select, 
                                     exclusion_colour = exclusion_colour)
        if select == 'inside':
            img = crop_image.crop_img_tl_br(img, tl_int, br_int)
        else:
            pass # it doesn't make sense to crop if the interest is on the rest of the image

        imgName = 'Press Enter to confirm'
        cv2.imshow(imgName, rescale_image.resize_max_pixel(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), max_pixel_rescale))
        key = cv2.waitKey(0)
        if key == ord('\r'): # enter key
            plt.close(fig)
            cv2.destroyWindow(imgName)
            break
        else:
            plt.close(fig)
            cv2.destroyWindow(imgName)
            continue

    if returnInt:
        tl = tl_int
        br = br_int
        coords_list = coords_list_int
    else: 
        pass
    return tl, br, coords_list

def manual_point_definition(img, dx = 100, dy = 100, max_pixel_rescale = 800,
                            color = (255,255,255), thickness = 2, returnInt = False):
    '''
    Given an image, it is possible to extract the coordinates of one pixel clicking on it

    Parameters
    ----------
    img : _type_
        _description_
    dx : int, optional
        _description_, by default 100
    dy : int, optional
        _description_, by default 100
    max_pixel_rescale : int, optional
        _description_, by default 800
    color : tuple, optional
        _description_, by default (255,255,255)
    thickness : int, optional
        _description_, by default 2

    Returns
    -------
    _type_
        _description_
    '''
    orig_img = img.copy()
    h, w = img.shape[:2]
    tl = [0, 0]
    br = [w, h]

    while True:
        img_for_drawing = orig_img.copy()
        fig, ax = plots_image.plot_image(orig_img, title = 'The second click will be considered. Use the first one to zoom')
        points = plt.ginput(n=2, timeout=-1, show_clicks=True)
        x = int(np.rint(points[-1][0]))
        y = int(np.rint(points[-1][1]))
        # just to draw the zoomed image
        tlx = int(np.rint(utils.scalar_in_range(x-dx,0,w)))
        tly = int(np.rint(utils.scalar_in_range(y-dy,0,h)))
        brx = int(np.rint(utils.scalar_in_range(x+dx,0,w)))
        bry = int(np.rint(utils.scalar_in_range(y+dy,0,h)))

        tl = [tlx, tly]
        br = [brx, bry]
        cv2.line(img_for_drawing, (0,y), (w,y), color, thickness) 
        cv2.line(img_for_drawing, (x,0), (x,h), color, thickness) 
        img_for_drawing = crop_image.crop_img_tl_br(img_for_drawing, tl, br)

        imgName = 'Press Enter to confirm'
        cv2.imshow(imgName, rescale_image.resize_max_pixel(cv2.cvtColor(img_for_drawing, cv2.COLOR_BGR2RGB), max_pixel_rescale))
        key = cv2.waitKey(0)
        if key == ord('\r'): # enter key
            plt.close(fig)
            cv2.destroyWindow(imgName)
            break
        else:
            plt.close(fig)
            cv2.destroyWindow(imgName)
            continue
    if returnInt:
        return int(np.round(x)), int(np.round(y))
    else:
        return x, y
    
def manual_crop_colour_description_tl_br(img_rgb, percentiles = [0.1, 0.25, 0.5, 0.75, 0.9], 
                                         max_pixel_rescale = 800, 
                                         show_projection = True, print_results = False):
    '''
    Given an RGB image, the user can crop a region using three points technique 
    and all the info  of that region are given


    Parameters
    ----------
    img_rgb : matrix width*height*3 or matrix width*height*1
        for a correct hsv conversion, the image should be rgb
    percentiles : list, optional
        list of percentiles of each channel to be analyzed, by default [0.1, 0.25, 0.5, 0.75, 0.9]
    print_results : boolean
        if printing tl br and description of rgb and hsv channels
        
    Returns
    -------
    tl : list of [x, y]
        top left coordinates
    br : list of [x, y]
        bottom right coordinates
    hProj : np.array
        mean of each row [array of height elements]
    vProj : np.array
        mean of each column [array of width elements]
    df_rgb: pandas df 
        containing the one column for each channel, one row for each value of that channel
    df_hsv : pandas df 
        containing the one column for each channel, one row for each value of that channel
    '''

    tl, br = get_coords_user_tl_br(img_rgb, nPoints = 3, max_pixel_rescale = max_pixel_rescale)
    img_cropped = crop_image.crop_img_tl_br(img_rgb, tl, br)
    hProj_rgb, vProj_rgb = info_image.projection(img_cropped, show = show_projection)

    df_rgb, desc_rgb = info_image.describe_channels(img_cropped, percentiles = percentiles)

    img_cropped_hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
    df_hsv, desc_hsv = info_image.describe_channels(img_cropped_hsv, percentiles = percentiles)
    hProj_hsv, vProj_hsv = info_image.projection(img_cropped_hsv, show = show_projection)

    if print_results:
        print('tl = {}\nbr = {}'.format(tl, br))
        print('RGB channel')
        print(desc_rgb)
        print('HSV channel')
        print(desc_hsv)

    return tl, br, df_rgb, df_hsv, hProj_rgb, vProj_rgb, hProj_hsv, vProj_hsv

def manual_crop_colour_description_contour(img_rgb, percentiles = [0.1, 0.25, 0.5, 0.75, 0.9], 
                                           select = 'inside', exclusion_colour = (0, 0, 0),
                                           max_pixel_rescale = 800, show_projection = True, print_results = False):
    '''
    User can tap nPoints on the image, the consecutive points will be connected 
    to create a contour. tl and br coordinates will be also obtained from the series of points
    
    
    Parameters
    ----------
    img_rgb : matrix width*height*3 or matrix width*height*1
        for a correct hsv conversion, the image should be rgb
    percentiles : list, optional
        list of percentiles of each channel to be analyzed, by default [0.1, 0.25, 0.5, 0.75, 0.9]
    print_results : boolean
        if printing tl br and description of rgb and hsv channels
        
    Returns
    -------
    tl : list of [x, y]
        top left coordinates
    br : list of [x, y]
        bottom right coordinates
    hProj : np.array
        mean of each row [array of height elements]
    vProj : np.array
       mean of each column [array of width elements]
    df_rgb: pandas df 
        containing the one column for each channel, one row for each value of that channel
    df_hsv : pandas df 
        containing the one column for each channel, one row for each value of that channel
    '''
    
    tl, br, coords = get_coords_user_contour(img_rgb, select = select, 
                                             exclusion_colour = exclusion_colour,
                                             max_pixel_rescale = max_pixel_rescale)
    contour = np.array(coords)
    selected_pixels_rgb = filter_image.get_selected_pixels_from_contour(img_rgb, contour)

    sub_rgb, mask, result = filter_image.filter_image_3ch(selected_pixels_rgb, ch0 = [-0.1, 0.1], ch1 = [-0.1, 0.1], ch2 = [-0.1, 0.1], 
                     trueValueFin = selected_pixels_rgb*np.nan,  falseValueFin = selected_pixels_rgb, 
                     show = False)
    img_cropped_rgb = crop_image.crop_img_tl_br(sub_rgb, tl, br)
    hProj_rgb, vProj_rgb = info_image.projection(img_cropped_rgb, show = show_projection)
    df_rgb, desc_rgb = info_image.describe_channels(img_cropped_rgb, percentiles = percentiles)
    
    selected_pixels_hsv = cv2.cvtColor(selected_pixels_rgb, cv2.COLOR_RGB2HSV)
    sub_hsv, mask, result = filter_image.filter_image_3ch(selected_pixels_hsv, ch0 = [-0.1, 0.1], ch1 = [-0.1, 0.1], ch2 = [-0.1, 0.1], 
                     trueValueFin = selected_pixels_hsv*np.nan,  falseValueFin = selected_pixels_hsv, 
                     show = False)
    img_cropped_hsv = crop_image.crop_img_tl_br(sub_hsv, tl, br)
    hProj_hsv, vProj_hsv = info_image.projection(img_cropped_hsv, show = show_projection)
    df_hsv, desc_hsv = info_image.describe_channels(img_cropped_hsv, percentiles = percentiles)
    
    if print_results:
        print('tl = {}\nbr = {}'.format(tl, br))
        print('RGB channel')
        print(desc_rgb)
        print('HSV channel')
        print(desc_hsv)

    return tl, br, df_rgb, df_hsv, hProj_rgb, vProj_rgb, hProj_hsv, vProj_hsv