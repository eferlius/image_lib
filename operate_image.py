import cv2
import numpy as np
from . import _check_
from . import plots_image

def erode(img, kernel, show = False, sharex = True, sharey = True,
                   nrows = 0, ncols = 0, mainTitle = None):
    img_ed = cv2.erode(img, kernel)
    if show:
        plots_image.plot_images_comparison(img, img_ed, sharex = sharex, sharey = sharey,
                           nrows = nrows, ncols = ncols, mainTitle = mainTitle)
    return img_ed

def dilate(img, kernel, show = False, sharex = True, sharey = True,
                   nrows = 0, ncols = 0, mainTitle = None):
    img_ed = cv2.dilate(img, kernel)
    if show:
        plots_image.plot_images_comparison(img, img_ed, sharex = sharex, sharey = sharey,
                   nrows = nrows, ncols = ncols, mainTitle = mainTitle)
    return img_ed

def erode_dilate(img, list_kernel_erosion, list_kernel_dilation, start_with = 'erosion', show = False, mainTitle = None):
    # EROSION_KERNEL = np.ones((5, 5), np.uint8) 
    # DILATATION_KERNEL = np.ones((15, 15), np.uint8)

    assert start_with in ['erosion', 'dilation'], "start_with should be either erosion or dilation, got {}".format(start_with)

    list_operations = [None]*(len(list_kernel_erosion)+len(list_kernel_dilation)) 
    list_kernels = [None]*(len(list_kernel_erosion)+len(list_kernel_dilation)) 
    if start_with == 'erosion':
        list_operations[::2] = ['erosion']*len(list_kernel_erosion)
        list_operations[1::2] = ['dilation']*len(list_kernel_dilation)
        list_kernels[::2] = list_kernel_erosion
        list_kernels[1::2] = list_kernel_dilation
    elif start_with == 'dilation':
        list_operations[::2] = ['dilation']*len(list_kernel_dilation)
        list_operations[1::2] = ['erosion']*len(list_kernel_erosion)
        list_kernels[::2] = list_kernel_dilation
        list_kernels[1::2] = list_kernel_erosion
    
    imagesDict = {}
    imagesDict['orig'] = img
    for operation, kernel, i in zip(list_operations, list_kernels, range(len(list_kernels))):
        if operation == 'erosion':
            img = cv2.erode(img, kernel) 
        elif operation == 'dilation':
            img = cv2.dilate(img, kernel) 
        imagesDict['{}: {}'.format(i, operation)] = img
    
    if show: 
        # call function for image show
        if not mainTitle:
            mainTitle = 'result of erosion and dilation'
        plots_image.plot_imagesDict_subplots(imagesDict,
        mainTitle = mainTitle)
        
        if not mainTitle:
            mainTitle = 'comparison of erosion and dilation process'
        plots_image.plot_images_comparison(imagesDict['orig'], img, mainTitle = mainTitle)

        # imagesDictComparison = {}
        # imagesDictComparison['orig'] = imagesDict['orig']
        # imagesDictComparison['final'] = img
        # comparison = img/2 - imagesDict['orig']/2
        # if len(comparison.shape) == 3:
        #     imagesDictComparison['comparison'] = comparison[:,:,0]
        # else:
        #     imagesDictComparison['comparison'] = comparison
            

        # plots_image.plot_imagesDict_subplots(imagesDictComparison,
        # mainTitle = mainTitle)

    return img
        
def erode_dilate_loop(img, kernel_erosion, kernel_dilation, n_loops = 3, start_with = 'erosion', show = False, mainTitle = None):
    return erode_dilate(img, [kernel_erosion]*n_loops, [kernel_dilation]*n_loops, start_with = start_with, show = show, mainTitle = mainTitle)


def rotate(img, angle, pivot, keep_whole_image = False, show = False):
    '''
    _summary_

    _extended_summary_

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        img to be rotated
    angle : float
        angle of rotation
    pivot : tuple of one string ([x, y])
        center of rotation coordinates
    keep_whole_image : bool, optional
        if True, makes the image bigger to show the whole rotated image
        if False, crops the rotated image to fit it into the original dimension, by default False
    show : bool, optional
        if showing the result, by default False

    Returns
    -------
    rotated: matrix width*height*3 or matrix width*height*1
        img to be rotated
    '''
    if keep_whole_image:
        padX = [img.shape[1] - pivot[0], pivot[0]]
        padY = [img.shape[0] - pivot[1], pivot[1]]
        if len(img.shape) == 2:
            imgP = np.pad(img, [padY, padX], 'constant')
        else:
            imgP = np.pad(img, [padY, padX, [0, 0]], 'constant')
        h, w = imgP.shape[:2]
        M = cv2.getRotationMatrix2D(([padX[0]+pivot[0], padY[0]+pivot[1]]), angle, 1.0)
        rotated = cv2.warpAffine(imgP, M, (w, h))
    else:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D(pivot, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
    if show: 
        plots_image.plot_image(rotated, title = 'rotated of {}°'.format(angle))
    return rotated