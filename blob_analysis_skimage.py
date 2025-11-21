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
    def __init__(self, img, method = 'DoH', **kwargs):
        '''
        Class of blob detection

        Parameters
        ----------
        img : matrix width*height*1
            image where the blobs_info will be computed
        method : str, optional
            method for blob extraction, can be:
            'LoG': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
            'DoG': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog
            'DoH': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh
            by default 'DoH'

        **kwargs are the arguments for the blob detection function, such as overlap and min_sigma, check documentation
        '''
        self.method = method
        if method == 'LoG':
            self.blobs_info = skimage.feature.blob_log(img, **kwargs)
        if method == 'DoG':
            self.blobs_info = skimage.feature.blob_dog(img, **kwargs)
        if method == 'DoH':
            self.blobs_info = skimage.feature.blob_doh(img, **kwargs)

        # swap x and y columns
            tmp = self.blobs_info[:,1].copy()
            self.blobs_info[:,1] = self.blobs_info[:,0]
            self.blobs_info[:,0] = tmp


        self.dictionary = self.create_dictionary()

    def create_dictionary(self):
        dictionary = {}
        dictionary['{}_xc'.format(self.method)] = self.blobs_info[:,0]
        dictionary['{}_yc'.format(self.method)] = self.blobs_info[:,1]
        dictionary['{}_std'.format(self.method)] = self.blobs_info[:,2]
        return dictionary      

def compute_detected_blobs_info_to_df(img, method = 'DoH', show = False, return_also_image = False, **kwargs_blob_det):
    '''
    Given a an image, compute the blobs_info
    Returns a pandas dataframe containing index, xc, yc and sigma of each blob

    Parameters
    ----------
    img : matrix width*height*1
        image where the blobs_info will be computed
    method : str, optional
        method for blob extraction, can be:
        'LoG': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
        'DoG': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog
        'DoH': https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh
        by default 'DoH'

    **kwargs are the arguments for the blob detection function, such as overlap and min_sigma, check documentation

    Returns
    -------
    Pandas dataframe containing index, xc, yc and sigma of each blob
    '''
    img_1ch = utils.convert_to_grayscale(img)

    output = detected_blobs_info(img_1ch, method = method, **kwargs_blob_det)
    df = pd.DataFrame(output.dictionary, columns = list(output.dictionary.keys()))
    df.insert(0, 'blob num', np.arange(0,df.shape[0]))
    if show or return_also_image:
        img_plot = draw_blobs_on_img(img, output.blobs_info)
        if show: 
            plots_image.plot_image(img_plot, 'blobs  with {} method'.format(method))
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






