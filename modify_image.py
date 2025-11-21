import cv2
import numpy as np
from . import utils

def validate_coordinates(tl_x, tl_y, br_x, br_y):
    """
    Validate and swap coordinates if necessary to ensure tl_x < br_x and tl_y < br_y.

    Parameters:
    - tl_x, tl_y: Top-left coordinates.
    - br_x, br_y: Bottom-right coordinates.

    Returns:
    - Tuple (tl_x, tl_y, br_x, br_y) with validated and possibly swapped coordinates.
    """
    if tl_x > br_x:
        tl_x, br_x = br_x, tl_x
    if tl_y > br_y:
        tl_y, br_y = br_y, tl_y

    return tl_x, tl_y, br_x, br_y

def color_pixels(orig_image, tl, br, colour=[0,0,0], in_out='out'):
    """
    Color pixels inside or outside a specified rectangle in an image.

    Parameters:
    - image: NumPy array representing the image.
    - tl: Tuple (x, y) representing the top-left coordinates of the rectangle.
    - br: Tuple (x, y) representing the bottom-right coordinates of the rectangle.
    - color: Tuple (B, G, R) representing the color to use for coloring pixels.
    - inside: Boolean flag. If True, color pixels inside the rectangle. If False, color pixels outside.

    Returns:
    - Modified image with pixels colored accordingly.
    """
    image = orig_image.copy()
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    tl_x, tl_y, br_x, br_y = validate_coordinates(tl[0], tl[1], br[0], br[1])

    # Check for NaN values in tl or br coordinates
    if np.isnan(tl_x) or np.isnan(tl_y) or np.isnan(br_x) or np.isnan(br_y):
        # If NaN values are present, color the entire frame
        image[:, :] = colour
    else:
        if in_out == 'in':
            mask[tl[1]:br[1] + 1, tl[0]:br[0] + 1] = 1
        elif in_out == 'out':
            mask[:tl_y, :] = 1
            mask[br_y + 1:, :] = 1
            mask[:, :tl_x] = 1
            mask[:, br_x + 1:] = 1

        image[mask == 1] = colour

    return image

def overlay_images(larger_image, smaller_image, top_left_x, top_left_y):
    # Extract the region of interest (ROI) from the larger image
    roi = larger_image[top_left_y:top_left_y+smaller_image.shape[0], top_left_x:top_left_x+smaller_image.shape[1]]
    roi = np.full_like(roi, [0]*utils.n_channels(roi), dtype=np.uint8)

    # Apply the smaller image to the ROI
    result_roi = cv2.add(roi, smaller_image)

    # Replace the original ROI in the larger image with the combined image
    larger_image[top_left_y:top_left_y+smaller_image.shape[0], top_left_x:top_left_x+smaller_image.shape[1]] = result_roi

    return larger_image