from . import plots_image

def crop_img_tl_br(img, tl, br, show = False, convertBGR2RGB = False):
    '''
    Crop image from top-left to bottom-right

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        image to be cropped
    tl : list of [x, y] coordinates
        top left coordinates.
    br : list of [x, y] coordinates
        bottom right coordinates.

    # example
    # tl = [686, 348]
    # br = [758, 388]

    Returns
    -------
    img : image: matrix
        cropped image.

    '''
    assert tl[0] < br[0] and tl[1] < br[1], ( 
        'not valid top-left/bottom-right coordinates. \ntl must be lower than br, got tl: {} and br: {}'.format(tl, br))

    try:
        img = img[tl[1]:br[1], tl[0]:br[0],:]
    except:
        img = img[tl[1]:br[1], tl[0]:br[0]]

    if show:
        plots_image.plot_image(img, title = 'cropped image from {} to {}'.format(tl, br), 
                  convertBGR2RGB = convertBGR2RGB)

    return img

def crop_img_n_regions(img, nrows = 2, ncols = 2, show = False):
    '''
    Crop image in N parts according to the number of rows and columns and returns a dictionary with:
    - indexes: r-c with r = row value and c = col value
    - values: the parts of the image 

    Parameters
    ----------
    img : matrix width*height*3 or matrix width*height*1
        _description_
    nrows : int, optional
        _description_, by default 2
    ncols : int, optional
        _description_, by default 2
    show : bool, optional
        _description_, by default False

    Returns
    -------
    dictionary
        _description_
    '''
    try:
        h, w, _ = img.shape
    except:
        h, w = img.shape
    imagesDict = {}
    for i in range(nrows):
        for j in range(ncols):
            try:
                imagesDict[str(i)+'-'+str(j)] = img[int(i*h/nrows):int((i+1)*h/nrows), int(j*w/ncols):int((j+1)*w/ncols),:]
            except:
                imagesDict[str(i)+'-'+str(j)] = img[int(i*h/nrows):int((i+1)*h/nrows), int(j*w/ncols):int((j+1)*w/ncols)]

    if show:
        # call function for image show
        plots_image.plot_imagesDict_subplots(imagesDict, nrows = nrows, ncols = ncols,
                              mainTitle = 'image cropped in {} row[s] * {} column[s]'.format(nrows, ncols))

    return imagesDict