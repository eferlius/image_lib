import cv2
def findCirclesOnImage(img, minDist, param1, param2, minRadius, maxRadius):
    '''
    find circles on the img, doesn't matter if RGB or BGR since it's converted 
    in gray scale. All the other parameters are the one of cv2.HoughCircles:
        https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Parameters
    ----------
    img : image
        where the circles should be detected.
    minDist : int
        between centers.
    param1 : int
        refer to documentation.
    param2 : int
        refer to documentation. In this case, param2 is recursively decreased 
        until when at least one circle is detected.
    minRadius : int
        minimum radius of the detected circles.
    maxRadius : int
        maximum radius of the detected circles.

    Returns
    -------
    circles : structure in the following shape:
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        containing the coordinates of centre and radius defining a circle

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = None
    coeff = 1
    # while loop changing param2: make the detection less picky till at least one circle is detected
    while circles is None:
        if coeff < 0:
            return None
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, \
                                   param1=param1, param2=param2 * coeff, minRadius=minRadius, maxRadius=maxRadius)
        coeff = coeff - 0.1 / coeff

    return circles