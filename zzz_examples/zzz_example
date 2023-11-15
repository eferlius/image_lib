import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(2, os.path.split(os.getcwd())[0])
import image_lib


SOURCE_PATH = r'G:\Shared drives\TableTennis\Tests\20230928\02_preprocessing\frames\mp_pose mc1_mdc0.5_mtc0.3\t01_side_ID8_pose'
FILENAME = r'f8575.png'

img = cv2.imread(os.path.join(SOURCE_PATH, FILENAME))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image_lib.user_interaction.manual_crop_colour_description_contour(img_rgb, print_results = True, exclusion_colour = (255, 0, 0))

# image_lib.filter_image.filter_image_3ch(img_rgb, ch0 = [151, 254], ch1 = [88, 194], ch2 = [39, 171], 
#                      trueValueFin = None,  falseValueFin = [0, 0, 0], 
#                      show = True)
# image_lib.filter_image.filter_image_3ch(img_rgb, ch0 = [188, 243], ch1 = [103, 194], ch2 = [56, 154], 
#                      trueValueFin = None,  falseValueFin = [0, 0, 0], 
#                      show = True)

# for the ball
image_lib.filter_image.filter_image_3ch(img_rgb, ch0 = [200, 243], ch1 = [103, 194], ch2 = [56, 154], 
                     trueValueFin = None,  falseValueFin = [0, 0, 0], 
                     show = True)

image_lib.filter_image.filter_image_3ch(img_rgb, ch0 = [25, 164], ch1 = [21, 163], ch2 = [15, 155], 
                     trueValueFin = None,  falseValueFin = [0, 0, 0], 
                     show = True)

plt.show()