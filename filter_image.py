import cv2
import numpy as np
from . import _check_
from . import plots_image

def get_imagesDict_basic_formats(img, imgFormat = 'BGR', show = False):
    '''
    Applies basic transformation on the input image and saves it in a dictionary.
    Operations are:
        - splitting in RGB
        - splitting in HSV
        - splitting in LAB
        - grayscale

    Parameters
    ----------
    img : matrix width*height*3
        assumed BGR, it's possible to specify it's RGB with imgFormat flag.
    imgFormat : string, optional
        image format, can be either BGR or RGB. The default is 'BGR'.
    show : bool, optional
        if showing the plot of the dictionary. The default is False.

    Returns
    -------
    imagesDict : dictionary
        contains as keys the name of the corresponding image.

    '''
    # check image format
    validImgFormats = ['BGR', 'RGB']
    assert imgFormat in validImgFormats, \
    f"imgFormat not valid, possible values are: {validImgFormats}"

    if imgFormat == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # create images dictionary
    imagesDict = {}
    imagesDict['RGB'] = img
    for i in range(3):
        imagesDict['RGB ch' + str(i)]=img[:,:,i]
    for i in range(3):
        imagesDict['HSV ch' + str(i)]=img_hsv[:,:,i]
    for i in range(3):
        imagesDict['HLS ch' + str(i)]=img_hls[:,:,i]
    for i in range(3):
        imagesDict['LAB ch' + str(i)]=img_lab[:,:,i]
    imagesDict['gray'] = img_gray

    # print(imagesDict.keys())
    if show:
        # call function for image show
        plots_image.plot_imagesDict_subplots(imagesDict, ncols = 4,
        mainTitle = 'image inspection on the different channels')

    return imagesDict

def filter_image_3ch(img, ch0 = [0, 255], ch1 = [0, 255], ch2 = [0, 255], 
                     trueValueFin = None,  falseValueFin = [0, 0, 0], 
                     show = False):
    '''
    Given an image, checks which pixels have value inside the ranges specified 
    in ch0, ch1 and ch2. 
    If ch[0] < ch[1], considers values from ch[0] to ch[1]
    If ch[0] > ch[1], considers values from min(ch) to ch[1] and from ch[0] to max(ch)

    Return three images:
    sub: image with falseValueFin (where conditions on ch0, ch1, ch2 are not met) 
    and trueValueFin (where conditions on ch0, ch1, ch2 are met)

    mask: image with 0 (where conditions on ch0, ch1, ch2 are not met) 
    and 255 (where conditions on ch0, ch1, ch2 are met)

    result: image with 0*3 (where conditions on ch0, ch1, ch2 are not met) 
    and 255*3 (where conditions on ch0, ch1, ch2 are met)

    Parameters
    ----------
    img : matrix width*height*3
        image to be filtered
    ch0 : list, optional
        lower and upper limit for channel 0, by default [0, 255]
    ch1 : list, optional
        lower and upper limit for channel 1, by default [0, 255]
    ch2 : list, optional
        lower and upper limit for channel 2, by default [0, 255]
    trueValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default None -> check substitute_values for more information
    falseValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default [0, 0, 0] -> check substitute_values for more information
    show : bool, optional
        if showing the plot of the dictionary. The default is False.

    Returns
    -------
    sub: image with falseValueFin (where conditions on ch0, ch1, ch2 are not met) 
    and trueValueFin (where conditions on ch0, ch1, ch2 are met)

    mask: image with 0 (where conditions on ch0, ch1, ch2 are not met) 
    and 255 (where conditions on ch0, ch1, ch2 are met)

    result: image with 0*3 (where conditions on ch0, ch1, ch2 are not met) 
    and 255*3 (where conditions on ch0, ch1, ch2 are met)
    '''

    true = img*0+255
    
    ch_lower = [ch0, ch1, ch2]
    ch_upper = [ch0, ch1, ch2]
    ch = [ch0, ch1, ch2]
    
    # if lower bound > upper bound, consider valid from min to upper bound and from lower bound to max
    for i in range(3):
        if ch[i][0]<ch[i][1]:
            pass
        else:
            ch_lower[i] = [np.min(img[:,:,i]), ch[i][1]]
            ch_upper[i] = [ch[i][0], np.max(img[:,:,i])]
    ch_lower = np.array(ch_lower).astype(type(np.array(ch)))
    ch_upper = np.array(ch_upper).astype(type(np.array(ch)))

    lower_ch_lower = np.array([ch_lower[0][0], ch_lower[1][0], ch_lower[2][0]])
    upper_ch_lower = np.array([ch_lower[0][1], ch_lower[1][1], ch_lower[2][1]])
    mask_lower = cv2.inRange(img, lower_ch_lower, upper_ch_lower)
    result_lower = cv2.bitwise_and(true, true, mask = mask_lower)
    
    lower_ch_upper = np.array([ch_upper[0][0], ch_upper[1][0], ch_upper[2][0]])
    upper_ch_upper = np.array([ch_upper[0][1], ch_upper[1][1], ch_upper[2][1]])
    mask_upper = cv2.inRange(img, lower_ch_upper, upper_ch_upper)
    result_upper = cv2.bitwise_and(true, true, mask = mask_upper)
    
    # merge the two results
    result = cv2.bitwise_or(result_lower, result_upper)
    mask = cv2.inRange(result, np.array([255,255,255]), np.array([255,255,255]))

    sub = substitute_values(img, mask, trueValueFin, falseValueFin, show = False)

    if show:
        imagesDict={}
        imagesDict['original'] = img
        imagesDict['sub: trueValue or falseValue'] = sub
        imagesDict['mask: 0 or 255'] = mask
        imagesDict['result: [0, 0, 0] or [255, 255, 255]'] = result

        plots_image.plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
        nrows = 2, ncols = 2, mainTitle = 'filtered with ' + str(ch0) + ' ' + str(ch1) + ' ' + str(ch2))
    
    return sub, mask, result

def filter_image_1ch(img, ch = [0, 255], trueValueFin = None,  falseValueFin = [0], 
                     show = False):
    '''
    Given an image, checks which pixels have value inside the ranges specified 
    in ch0, ch1 and ch2. 
    If ch[0] < ch[1], considers values from ch[0] to ch[1]
    If ch[0] > ch[1], considers values from min(ch) to ch[1] and from ch[0] to max(ch)

    Return three images:
    sub: image with falseValueFin (where conditions on ch are not met) 
    and trueValueFin (where conditions on ch are met)

    mask: image with 0 (where conditions on ch are not met) 
    and 255 (where conditions on ch are met)

    result: image with 0 (where conditions on ch are not met) 
    and 255 (where conditions on ch are met)

    Parameters
    ----------
    img : matrix width*height*1
        image to be filtered
    ch : list, optional
        lower and upper limit for channel 0, by default [0, 255]
    trueValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default None -> check substitute_values for more information
    falseValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default [0, 0, 0] -> check substitute_values for more information
    show : bool, optional
        if showing the plot of the dictionary. The default is False.

    Returns
    -------
    sub: image with falseValueFin (where conditions on ch are not met) 
    and trueValueFin (where conditions on ch are met)

    mask: image with 0 (where conditions on ch are not met) 
    and 255 (where conditions on ch are met)

    result: image with 0 (where conditions on ch are not met) 
    and 255 (where conditions on ch are met)
    '''

    true = img*0+255
    
    # if lower bound > upper bound, consider valid from min to upper bound and from lower bound to max
    if ch[0]<ch[1]:
        pass
    else:
        ch_lower = [np.min(img[:,:]), ch[1]]
        ch_upper = [ch[0], np.max(img[:,:])]

    mask_lower = cv2.inRange(img, np.array(ch_lower[0]).astype(int), np.array(ch_lower[1]).astype(int))
    result_lower = cv2.bitwise_and(true, true, mask = mask_lower)
    
    mask_upper = cv2.inRange(img,  np.array(ch_upper[0]).astype(int), np.array(ch_upper[1]).astype(int))
    result_upper = cv2.bitwise_and(true, true, mask = mask_upper)
    
    # merge the two results
    result = cv2.bitwise_or(result_lower, result_upper)
    mask = cv2.inRange(result, np.array([255]), np.array([255]))

    sub = substitute_values(img, mask, trueValueFin, falseValueFin, show = False)

    if show:
        imagesDict={}
        imagesDict['original'] = img
        imagesDict['sub: trueValue or falseValue'] = sub
        imagesDict['mask: 0 or 255'] = mask
        imagesDict['result: 0 or 255'] = result

        plots_image.plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
        nrows = 2, ncols = 2, mainTitle = 'filtered with ' + str(ch))
    
    return sub, mask, result
    

def substitute_values(img, mask, trueValueFin = None, falseValueFin = None, show = False):
    '''
    Given an image and a mask, substitute trueValueFin where mask is true and 
    falseValueFin where mask is false.

    if both trueValueFin and falseValueFin are None, returns an img where mask is True and 0 (or 0*3) where mask is False
    if one of the two is a given colour (expressed in the same number of channels of input image),
    then that colour is used for either true values or false values and the other is the original img

    if one of the two is a given image (expressed in the same number of channels of input image),
    then that image is used for either true values or false values and the other is the original img

    two colours or two images can be combined in trueValueFin and falseValueFin as long as they have the same depth

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to be substituted
    mask : matrix width*height*1
        mask for filtering
    trueValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default None
    falseValueFin : None or int or list or matrix width*height*1 or matrix width*height*3, optional
        values to give to sub where the result of filtering are true, 
        by default None
    show : bool, optional
        if showing the plot of the comparison of original image and substituted image. The default is False

    Returns
    -------
    sub : substituted image
    '''
    if trueValueFin is None and falseValueFin is None:
        sub = cv2.bitwise_and(img, img, mask = mask)
    else:
        shape = img.shape
        if len(shape) == 3:
            h, w, _ = shape
            # expand dimension of mask
            mask_exp = np.expand_dims(mask, axis = -1)
        elif len(shape) == 2:
            h, w = shape
            mask_exp = mask
        
        if trueValueFin is not None:
            if np.isscalar(trueValueFin):
                trueValueFin = img*0+trueValueFin
            elif ((_check_.is_list(trueValueFin) or _check_.is_npArray(trueValueFin)) and len(trueValueFin) == 3):
                trueValueFin = np.full((h, w, 3), trueValueFin, dtype=np.uint8)
        else:
            trueValueFin = img
                
        if falseValueFin is not None:
            if np.isscalar(falseValueFin):
                falseValueFin = img*0+falseValueFin
            elif ((_check_.is_list(falseValueFin) or _check_.is_npArray(falseValueFin)) and len(falseValueFin) == 3):
                falseValueFin = np.full((h, w, 3), falseValueFin, dtype=np.uint8)
        else:
            falseValueFin = img

        # if mask_exp has depth 1 while trueValueFin and falseValueFin have depth 3: expand mask
        if (len(mask_exp.shape) == 2 and len(trueValueFin.shape) == 3 and len(falseValueFin.shape) == 3):
            mask_exp = np.expand_dims(mask_exp, axis = -1)
                
        sub = np.where(mask_exp, trueValueFin, falseValueFin)
        
    if show:
        imagesDict={}
        imagesDict['orig'] = img
        imagesDict['sub'] = sub
        plots_image.plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
        nrows = 0, ncols = 1)
        
    return sub

def get_selected_pixels_from_contour(img, contour, select = 'inside', 
                                     exclusion_colour = (0, 0, 0), show = False):
    '''
    Given an image and a list of coordinates as contour, extracts the image
    inside or outside (according to select) the contour and makes of exclusion_colour
    the rest of th eimage

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image whose region wants to be isolated
    contour : np.array containing coordinates of points
        np.array([[x0, y0], [x1, y1], ...])
    select : str, optional
        if considering valid the region 'inside' or 'outside' the contour, by default 'inside'
    exclusion_colour : tuple, optional
        colour of the exclusion region, by default (0, 0, 0)
    show : bool, optional
        if showing the plot of the comparison of original image and substituted image. The default is False.

    Returns
    -------
    res : image with original values where specified by contour and exclusion_colour elsewhere
    '''
    # Create a blank mask
    mask = np.zeros_like(img, dtype=np.uint8)
    # Draw the filled contour on the mask
    cv2.drawContours(mask, [contour], 0, (255,255,255), thickness=cv2.FILLED)
    if select == 'inside':
        pass
    if select == 'outside':
        mask = cv2.bitwise_not(mask)
    # Use the mask to extract the selected pixels from the original image
    res = np.where(mask, img, exclusion_colour)

    if show:
        imagesDict={}
        imagesDict['original'] = img
        imagesDict['contour'] = res
        plots_image.plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
        nrows = 2, mainTitle = 'contours {}'.format(select))

    return res.astype(np.uint8)

def extract_Kmeans(img, n_clusters = 3, highlightValue = [0,0,0], 
                   criteria_iterations = 10, attempts = 10, show = False):
    '''
    Divides the image in n_clusters according to the features given

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to be clustered.
    n_clusters : int, optional
        number of clusters. The default is 3.
    highlightValue : scalar of list of 3 channels values, optional
        among all the centers of clusters, the closest to this value will be considered valid. 
        The default is [0,0,0].
    show : bool, optional
        if showing the plot of the comparison of original image, clustered image and mask. The default is False.
    criteria_iterations : int, optional
        number of iterations in the iteration criteria. The default is 10.
    attempts : int, optional
        number of times the algorithm is executed using different initial labellings. The default is 10.

    Raises
    ------
    Exception
        if show is true and nFeatures is bigger than 3.

    Returns
    -------
    segmentedImg : matrix width*height*3 or matrix width*height*1
        image with clustered colours.
    highlightImg :  matrix width*height*3 or matrix width*height*1
        original image with pixels close to highlightValue are of highlightValue colour.
    mask :  matrix width*height*3 or matrix width*height*1
        0 where pixel is not in highlightValue cluster
        1 where pixel is in highlightValue cluster
    '''
    if len(img.shape) == 2:
        nFeatures = 1
    else:
        nFeatures = img.shape[-1]

    if nFeatures > 3 and show:
        raise Exception("can't display image if nFeatures > 3")

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, nFeatures))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, criteria_iterations, 1.0)
    _, labels, (centers) = cv2.kmeans(pixel_values, n_clusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.rint(centers)
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmentedImg = centers[labels]
    # reshape back to the original image dimension
    segmentedImg = segmentedImg.reshape(img.shape)
    
    highlightValue = np.array(highlightValue)
    norms = []
    for center in centers:
        norms.append(np.linalg.norm(center - highlightValue))
    
    # find the index of the closest center to highlightValue
    highlightIndex = np.argmin(norms)
    
    if len(img.shape) == 2:
        mask = (np.array(segmentedImg == centers[highlightIndex])*1).astype(int)
    else:
        mask = (np.array(segmentedImg == centers[highlightIndex]).all(axis = 2)*1).astype(int)
        
    highlightImg = substitute_values(img, mask, trueValueFin = highlightValue)
    
    if show:
        imagesDict={}
        imagesDict['original'] = img
        imagesDict['segmented'] = segmentedImg
        imagesDict['highlighted'] = highlightImg
        imagesDict['mask'] = mask
        plots_image.plot_imagesDict_subplots(imagesDict, sharex = True, sharey = True,
        nrows = 2, ncols = 2, mainTitle = 'extracted with kmeans - {} clusters'.format(n_clusters))
        # plots_image.plot_image(segmentedImg, title = 'extracted with kmeans - {} clusters'.format(n_clusters))
        # plots_image.plot_image(highlightImg, title = 'extracted with kmeans - highlighted cluster')

    return segmentedImg, highlightImg, mask

