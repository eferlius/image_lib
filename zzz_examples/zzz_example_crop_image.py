import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(2, os.path.split(os.getcwd())[0])
import image_lib

# Load your image
image = cv2.imread('retina.png')

# print(image_lib.user_interaction.get_coords_user_tl_br(image, nPoints = 3, returnInt = True))
print(image_lib.user_interaction.get_coords_user_contour(image, select = 'outside', exclusion_colour = (200, 0, 100), returnInt = True))

# image_lib.user_interaction.manual_crop_colour_description_contour(image)

contour = np.array([[500, 500],[800, 500], [700, 700]])

image_lib.filter_image.get_selected_pixels_from_contour(image, contour, 'inside', (200, 100, 50), show = True)
image_lib.filter_image.get_selected_pixels_from_contour(image, contour, 'outside', (200, 100, 50), show = True)
image_lib.filter_image.get_selected_pixels_from_contour(image[:,:,0], contour, 'inside', (0), show = True)

# tl, br, coords = image_lib.coords_on_image.get_coords_user_contour(image)
# print(coords)

# # Get the contour from your PyQt application
# contour = np.array(coords)

# # Get the selected pixels using the function
# selected_pixels = image_lib.filter_image.get_selected_pixels(image, contour)

# # Now 'selected_pixels' contains the values of the pixels inside the user-drawn shape
# image_lib.plots_image.plot_image(selected_pixels)

# # nan replacing for mean computation
# sub, mask, result = image_lib.filter_image.filter_image_3ch(selected_pixels, ch0 = [0, 1], ch1 = [0, 1], ch2 = [0, 1], 
#                      trueValueFin = selected_pixels*np.nan,  falseValueFin = selected_pixels, 
#                      show = True)

# df, desc = image_lib.info_image.describe_channels(sub)
# print(desc)

plt.show()

