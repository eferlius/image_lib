"""
wrapper for _plots_image_ taken from basic v0/basic_plot
"""
import cv2
from . import _plots_image_

def plot_image(img, convertBGR2RGB = False, mainTitle = ''):
    '''
    Shows an image in a matplotlib figure
    MATPLOTLIB wants RGB image
    CV2 wants BGR image
    '''
    if convertBGR2RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        pass
    fig, ax = _plots_image_.plts_img(img, mainTitle = mainTitle)
    return fig, ax

def plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
                   nrows = 0, ncols = 0, mainTitle = ''):
    '''
    From dictionary of images calls _plots_image_.plts_img and shows every image
    '''
    fig, ax = _plots_image_.plts_img(list(imagesDict.values()), 
                                     listTitles = list(imagesDict.keys()), 
                                     sharex = sharex, sharey = sharey, 
                                     nrows = nrows, ncols = ncols, 
                                     mainTitle = mainTitle)

    return fig, ax

def plot_images_comparison(original, final, sharex = True, sharey = True,
                   nrows = 0, ncols = 0, mainTitle = ''):
    imagesDictComparison = {}
    imagesDictComparison['orig'] = original
    imagesDictComparison['final'] = final
    comparison = final/2 - original/2
    if len(comparison.shape) == 3:
        imagesDictComparison['comparison'] = comparison[:,:,0]
    else:
        imagesDictComparison['comparison'] = comparison
    return plot_imagesDict_subplots(imagesDictComparison, sharex = sharex, sharey = sharey,
                   nrows = nrows, ncols = ncols, mainTitle = mainTitle)

