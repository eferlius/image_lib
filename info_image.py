import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from . import crop_image
from . import coords_on_image
from . import filter_image

def projection(img, show = False):
    '''
    horizontal and vertical projection of the image
    
    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image whose channels want to be analyzed
    show : bool, optional
        if showing the plot of the projections of the image. The default is False.
        

    Returns
    -------
    hProj : np.array
        mean of each row [array of height elements]
    vProj : np.array
       mean of each column [array of width elements]
    '''
    # represent each row with the mean value [array of height elements]
    hProj = np.nanmean(img, axis=1)
    # represent each column with the mean value [array of width elements]
    vProj = np.nanmean(img, axis=0)

    if show:
        fig = plt.figure()
        ax2 = fig.add_subplot(222)
        ax1 = fig.add_subplot(221, sharey = ax2)
        ax3 = fig.add_subplot(224, sharex = ax2)

        ax1.plot(hProj,np.arange(0,len(hProj)),'.-')
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_title('rows [hProj]')

        ax2.imshow(img, aspect="auto")

        ax3.plot(vProj,'.-')
        ax3.set_title('cols [vProj]')

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)

    return hProj, vProj

def describe_channels(img, percentiles = None):
    '''
    Given an image, creates a pandas dataframe for each channel and returns the 
    description with mean, median, std deviation and percentiles of each channel

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image whose channels want to be analyzed
    percentiles : list of values between 0 and 1 or None, optional
        percentiles to be computed, by default None

    Returns
    -------
    df: pandas df 
        containing the one column for each channel, one row for each value of that channel
    desc: pandas df 
        containing df.describe() of each channel (mean, median, std deviation and percentiles)
    '''
    valuesDict = {}
    transposed = np.transpose(img)

    for i in range(transposed.shape[0]):
        valuesDict['ch{}'.format(i)] = transposed[i].flatten()

    df = pd.DataFrame.from_dict(valuesDict)
    desc = df.describe(percentiles)

    return df, desc


