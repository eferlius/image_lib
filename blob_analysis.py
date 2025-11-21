import pandas as pd
import cv2
import numpy as np
from . import plots_image
from . import utils

"""
date: 2024-06-19 11:38:55
note: use contour
contours, hierarcy = cv2.findContours(frame[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
"""

def compute_line_inside_img(x, m, q, miny, maxy):
    y = m*x+q
    if y > miny and y < maxy:
        return x, y
    if y > maxy:
        y = maxy
    elif y < miny:
        y = miny
    x = (y-q)/m
    return x, y
             
class blob_properties:
    def __init__(self, contour = None, img = None):

        self.simple_blob = simple_blob(contour)
        self.contour_area = cv2.contourArea(contour)
        self.contour_perimeter = cv2.arcLength(contour,True) # boolean if closed contour perimeter
        self.contour_compactness  = self.contour_area / self.contour_perimeter
        self.convexity = cv2.isContourConvex(contour)

        # bounding rectangle
        self.bound_rect = bounding_rectangle(contour)
        self.extent = float(self.contour_area)/self.bound_rect.area

        # minimum area rectangle
        self.min_area_rect = minimum_area_rectangle(contour)

        # hull shape
        self.hull_shape = hull_shape(contour)
        if self.hull_shape.area == 0:
            self.solidity = None
        else:
            self.solidity = float(self.contour_area)/self.hull_shape.area

        # ellipse
        self.ellipse = ellipse(contour)
        
        # minimum enclosing circle
        self.min_enc_circle = minimum_enclosing_circle(contour)

        # line fitting
        self.fit_line = fitting_line(contour, img)

        self.dictionary = self.create_dictionary()

    def create_dictionary(self):
        dictionary = {}
        dictionary['simple_blob_xc'] = self.simple_blob.xc
        dictionary['simple_blob_yc'] = self.simple_blob.yc

        dictionary['contour_area'] = self.contour_area
        dictionary['contour_perimeter'] = self.contour_perimeter
        dictionary['contour_compactness'] = self.contour_compactness

        dictionary['convexity'] = self.convexity

        dictionary['bound_rect_x0'] = self.bound_rect.x0
        dictionary['bound_rect_y0'] = self.bound_rect.y0
        dictionary['bound_rect_x1'] = self.bound_rect.x1
        dictionary['bound_rect_y1'] = self.bound_rect.y1
        dictionary['bound_rect_w'] = self.bound_rect.w
        dictionary['bound_rect_h'] = self.bound_rect.h
        dictionary['bound_rect_aspect_ratio'] = self.bound_rect.aspect_ratio
        dictionary['bound_rect_area'] = self.bound_rect.area
        dictionary['extent'] = self.extent

        dictionary['min_area_rect_xc'] = self.min_area_rect.xc
        dictionary['min_area_rect_yc'] = self.min_area_rect.yc
        dictionary['min_area_rect_w'] = self.min_area_rect.w
        dictionary['min_area_rect_h'] = self.min_area_rect.h
        dictionary['min_area_rect_theta'] = self.min_area_rect.theta
        dictionary['min_area_rect_xtr'] = self.min_area_rect.xtr
        dictionary['min_area_rect_ytr'] = self.min_area_rect.ytr
        dictionary['min_area_rect_xtl'] = self.min_area_rect.xtl
        dictionary['min_area_rect_ytl'] = self.min_area_rect.ytl
        dictionary['min_area_rect_xbr'] = self.min_area_rect.xbr
        dictionary['min_area_rect_ybr'] = self.min_area_rect.ybr
        dictionary['min_area_rect_xbl'] = self.min_area_rect.xbl
        dictionary['min_area_rect_ybl'] = self.min_area_rect.ybl

        dictionary['hull_area'] = self.hull_shape.area
        dictionary['solidity'] = self.solidity

        dictionary['ellipse_xc'] = self.ellipse.xc
        dictionary['ellipse_yc'] = self.ellipse.yc
        dictionary['ellipse_major_axis'] = self.ellipse.major_axis
        dictionary['ellipse_minor_axis'] = self.ellipse.minor_axis
        dictionary['ellipse_theta'] = self.ellipse.theta
        dictionary['ellipse_aspect_ratio'] = self.ellipse.aspect_ratio
 
        dictionary['min_enc_circle_xc'] = self.min_enc_circle.xc
        dictionary['min_enc_circle_yc'] = self.min_enc_circle.yc
        dictionary['min_enc_circle_r'] = self.min_enc_circle.radius

        dictionary['fit_line_m'] = self.fit_line.m
        dictionary['fit_line_q'] = self.fit_line.q
        dictionary['fit_line_xf'] = self.fit_line.xf
        dictionary['fit_line_yf'] = self.fit_line.yf
        dictionary['fit_line_xl'] = self.fit_line.xl
        dictionary['fit_line_yl'] = self.fit_line.yl
        dictionary['fit_line_xr'] = self.fit_line.xr
        dictionary['fit_line_yr'] = self.fit_line.yr

        return dictionary 
    
class simple_blob:
    def __init__(self, contour):
        # Calculate the moments
        moments = cv2.moments(contour)

        # Calculate the centroid
        if moments['m00'] != 0:
            self.xc = int(moments['m10'] / moments['m00'])
            self.yc = int(moments['m01'] / moments['m00'])
        else:
            self.xc = None
            self.yc = None
            
    def draw_on_img(self, img, colour = None):
        if not self.xc or not self.yc:
            pass
        else:
            if not colour:
                colour = utils.colour_according_to_img(img)
            cv2.circle(img, (self.xc, self.yc), radius=3, color=colour, thickness=-1)

class bounding_rectangle:
    def __init__(self, contour):
        self.x0, self.y0, self.w, self.h =  cv2.boundingRect(contour)
        self.x1 = self.x0 + self.w
        self.y1 = self.y0 + self.h
        self.aspect_ratio = self.w / self.h
        self.area = self.w * self.h
    def draw_on_img(self, img, colour = None, thickness = 2):
        if not colour:
            colour = utils.colour_according_to_img(img)
        cv2.rectangle(img, (self.x0, self.y0), (self.x1, self.y1), colour, thickness)

class minimum_area_rectangle:
    def __init__(self, contour):
       self.rect = cv2.minAreaRect(contour)
       (self.xc, self.yc), (self.w, self.h), self.theta = self.rect
       self.box = cv2.boxPoints(self.rect)
       self.xtr = self.box[0,0]
       self.ytr = self.box[0,1]
       self.xtl = self.box[1,0]
       self.ytl = self.box[1,1]
       self.xbr = self.box[2,0]
       self.ybr = self.box[2,1]
       self.xbl = self.box[3,0]
       self.ybl = self.box[3,1]
    def draw_on_img(self, img, colour = None, thickness = 2):
        if not colour:
            colour = utils.colour_according_to_img(img)
        cv2.drawContours(img, np.int0([self.box]), 0, colour, thickness)

class hull_shape:
    def __init__(self, contour):
        self.hull = cv2.convexHull(contour) # is a list of points
        self.area = cv2.contourArea(self.hull)

class ellipse:
    def __init__(self, contour):
        (self.xc, self.yc), (self.major_axis, self.minor_axis), self.theta = cv2.fitEllipse(contour)
        self.aspect_ratio = self.major_axis / self.minor_axis
    def draw_on_img(self, img, colour = None, thickness = 2):
        if not colour:
            colour = utils.colour_according_to_img(img)
        cv2.ellipse(img, ((self.xc, self.yc), (self.major_axis, self.minor_axis), self.theta), colour, thickness)

class minimum_enclosing_circle:
    def __init__(self, contour):
        (self.xc, self.yc), self.radius = cv2.minEnclosingCircle(contour)
    def draw_on_img(self, img, colour = None, thickness = 2):
        if not colour:
            colour = utils.colour_according_to_img(img)
        cv2.circle(img, (int(self.xc), int(self.yc)), int(self.radius), colour, thickness)

class fitting_line:
    def __init__(self, contour, img):
        [self.vx, self.vy, self.xf, self.yf] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
        self.xf = (self.xf)[0] #otherwise returns array with one value
        self.yf = (self.yf)[0] #otherwise returns array with one value
        self.m = (self.vy / self.vx)[0] #otherwise returns array with one value
        self.q = self.yf - self.m * self.xf #otherwise returns array with one value
        
        rows,cols = img.shape[:2]
        self. xl, self.yl = compute_line_inside_img(0, self.m, self.q, 0, rows-1)
        self. xr, self.yr = compute_line_inside_img(cols-1, self.m, self.q, 0, rows-1)
        self.xl = int(self.xl)
        self.yl = int(self.yl)
        self.xr = int(self.xr)
        self.yr = int(self.yr)
    def draw_on_img(self, img, colour = None, thickness = 2):
        if not colour:
            colour = utils.colour_according_to_img(img)
        cv2.line(img, (self.xl, self.yl), (self.xr, self.yr), colour, thickness)

def compute_blob_properties_to_obj(contour, img):
    '''
    Given a contour and the image where it was extracted, computes the blob properties
    Simply calls the __init__ function of the class blob_properties

    Parameters
    ----------
    contour : array or list of coordinates [[[x0,y0]],[[x1,y1]]...[[xn,yn]]]
        points defining the contour
    img : matrix width*height*3 or matrix width*height*1
        image where the contours were computed

    Returns
    -------
    blob_properties object with all the attributes of the blob associated to that contour
    '''
    return blob_properties(contour, img)

def compute_blob_properties_to_dict(contour, img):
    '''
    Given a contour and the image where it was extracted, computes the blob properties
    Simply calls the __init__ function of the class blob_properties

    Parameters
    ----------
    contour : array or list of coordinates [[[x0,y0]],[[x1,y1]]...[[xn,yn]]]
        points defining the contour
    img : matrix width*height*3 or matrix width*height*1
        image where the contours were computed

    Returns
    -------
    blob_properties dictionary with all the attributes of the blob associated to that contour
    '''
    # just a reminder
    # list(compute_blob_properties_to_dict(contour, img).keys())
    # list(compute_blob_properties_to_dict(contour, img).values())
    return blob_properties(contour, img).dictionary

def convert_blob_properties_obj_to_dict(blob_properties_obj):
    return blob_properties_obj.dictionary

def compute_blobs_properties_to_df(contours, img, show = False, return_also_image = False):
    '''
    Given a list of contours, for each one computes the blob properties
    Returns a pandas dataframe containing all the properties of each blob

    Parameters
    ----------
    contours : list of contours
        contours, hierarcy = cv2.findContours(img[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_plot = cv2.drawContours(image_plot, contours, -1, (0, 0, 255), 3)
    img : matrix width*height*3 or matrix width*height*1
        image where the contours were computed

    Returns
    -------
    Pandas dataframe containing all the properties of each blob
    '''
    rows = [] # for the dataframe
    rows_obj = [] # for drawing
    for contour in contours:
        try:
            output = compute_blob_properties_to_obj(contour, img)
            rows_obj.append(output)
            rows.append(list(output.dictionary.values()))
        except:
            pass
    df = pd.DataFrame(rows, columns = list(output.dictionary.keys()))
    df.insert(0, 'blob num', np.arange(0,df.shape[0]))
    if show or return_also_image:
        img_plot = draw_blobs_on_img(img, rows_obj)
        if show: 
            plots_image.plot_image(img_plot, 'blobs detected')
        if return_also_image:
            return df, img_plot
    return df

def compute_blobs_properties_to_list_of_obj(contours, img, show = False, return_also_image = False):
    '''
    Given a list of contours, for each one computes the blob properties
    Returns a list of blob_properties objects

    Parameters
    ----------
    contours : list of contours
        contours, hierarcy = cv2.findContours(img[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_plot = cv2.drawContours(image_plot, contours, -1, (0, 0, 255), 3)
    img : matrix width*height*3 or matrix width*height*1
        image where the contours were computed

    Returns
    -------
    list of blob_properties object with all the attributes of the blob associated to that contour
    '''
    rows_obj = []
    for contour in contours:
        try:
            rows_obj.append(compute_blob_properties_to_obj(contour, img))
        except:
            pass
    if show or return_also_image:
        img_plot = draw_blobs_on_img(img, rows_obj)
        if show: 
            plots_image.plot_image(img_plot, 'blobs detected')
        if return_also_image:
            return rows_obj, img_plot
    return rows_obj

def convert_blobs_properties_list_of_obj_to_df(list_of_obj):
    rows = [] # for the dataframe
    for obj in list_of_obj:
        try:
            rows.append(list(obj.dictionary.values()))
        except:
            pass
    df = pd.DataFrame(rows, columns = list(obj.dictionary.keys()))
    df.insert(0, 'blob num', np.arange(0,df.shape[0]))
    return df

def draw_blob_on_img(img, blob_properties_obj, colour = None, thickness = 2):
    '''
    draws on the image
    - bounding rectangle, 
    - minimum enclosing rectangle, 
    - ellipse, 
    - minimum enclosing circle, 
    - fitting line
    of the given blob, expressed with a blob_properties class

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to draw on
    blob_properties_obj : blob_properties class
        blob_properties object with all the attributes of the blob associated to that contour
    colour : _type_, optional
        _description_, by default None
    thickness : int, optional
        _description_, by default 2

    Returns
    -------
    img with drawings on it
    '''
    if colour == None:
        colour0 = [200, 10, 10]
        colour1 = [10, 200, 10]
        colour2 = [10, 10, 200]
        colour3 = [200, 10, 200]
        colour4 = [200, 200, 10]
        colour5 = [10, 200, 200]
        colour = [colour0, colour1, colour2, colour3, colour4, colour5]
        
    img_copy = img.copy()
    blob_properties_obj.bound_rect.draw_on_img(img_copy, colour[0], thickness)
    blob_properties_obj.min_area_rect.draw_on_img(img_copy, colour[1], thickness)
    blob_properties_obj.ellipse.draw_on_img(img_copy, colour[2], thickness)
    blob_properties_obj.min_enc_circle.draw_on_img(img_copy, colour[3], thickness)
    blob_properties_obj.fit_line.draw_on_img(img_copy, colour[4], int(thickness/2))
    blob_properties_obj.simple_blob.draw_on_img(img_copy, colour[5])
    return img_copy

def draw_blobs_on_img(img, list_of_blob_properties_obj, colour = None, thickness = 2):
    '''
    draws on the image
    - bounding rectangle, 
    - minimum enclosing rectangle, 
    - ellipse, 
    - minimum enclosing circle, 
    - fitting line
    of all the blob_properties classes in the list

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to draw on
    list_of_blob_properties_obj : _type_
        _description_
    colour : _type_, optional
        _description_, by default None
    thickness : int, optional
        _description_, by default 2

    Returns
    -------
    img with drawings on it
    '''
    img_copy = img.copy()
    i = -1
    for blob_properties_obj in list_of_blob_properties_obj:
        i+=1
        img_copy = draw_blob_on_img(img_copy, blob_properties_obj, colour, thickness)
        cv2.putText(img_copy, str(i), (int(blob_properties_obj.fit_line.xf), int(blob_properties_obj.fit_line.yf)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], thickness, cv2.LINE_AA)
    return img_copy


