import numpy as np
import cv2
import sys
import os
sys.path.insert(1, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(2, os.path.split(os.getcwd())[0])
import image_lib

import matplotlib.pyplot as plt

# Create a 256x256x3 array using the gradients for each channel
image_data = np.zeros((256, 256, 3), dtype=np.uint8)
image_data[:, :, 0] = np.linspace(0, 255, 256, dtype=np.uint8)[np.newaxis, :]
image_data[:, :, 1] = np.linspace(0, 255, 256, dtype=np.uint8)[:, np.newaxis]
image_data[:, :, 2] = np.zeros((256, 256), dtype=np.uint8) + 255
img = image_data

image_data = np.zeros((256, 256, 3), dtype=np.uint8)
image_data[:, :, 0] = np.linspace(255, 0, 256, dtype=np.uint8)[np.newaxis, :]
image_data[:, :, 1] = np.linspace(255, 0, 256, dtype=np.uint8)[:, np.newaxis]
image_data[:, :, 2] = np.zeros((256, 256), dtype=np.uint8) + 255
img2 = image_data



filt, mask, result = image_lib.filter_image.filter_image_3ch(img, [160, 40], [150, 50], [0, 255], 
                                                trueValueFin = None, show = True)

# repeating a 1 channel image 3 times to use the filter_3ch
merged_img = cv2.merge((img[:,:,0], img[:,:,0], img[:,:,0]))
filt, mask, result = image_lib.filter_image.filter_image_3ch(merged_img, [160, 40], [160, 40], [160, 40], 
                                                trueValueFin = None, show = True)

# using fliter_1ch on a 1 channel image
filt, mask, result = image_lib.filter_image.filter_image_1ch(img[:,:,0], [160, 40], 
                                                trueValueFin = None, falseValueFin = img[:,:,1], show = True)

# trueValueFin and falseValueFin are None and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = None, falseValueFin = None, show = True)
# trueValueFin is a given 3d colour and falseValueFin is None and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = [255, 0, 0], falseValueFin = None, show = True)
# trueValueFin is None and falseValueFin is a given 3d colour and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = None, falseValueFin = [255, 0, 0], show = True)
# trueValueFin and falseValueFin are 3d coloursr and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = [255, 0, 0], falseValueFin = [255, 0, 255], show = True)
# trueValueFin and falseValueFin are 3d images and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = img, falseValueFin = img2, show = True)
# trueValueFin is 3d colour and falseValueFin is 3d image and img is 3d
image_lib.filter_image.substitute_values(img, mask, trueValueFin = [200, 0, 0], falseValueFin = img2, show = True)
# trueValueFin is 3d colour and falseValueFin is 3d image and img is 1d
image_lib.filter_image.substitute_values(img[:,:,0], mask, trueValueFin = [200, 0, 0], falseValueFin = img2, show = True)
# trueValueFin is 1d colour and falseValueFin is 1d image and img is 1d
image_lib.filter_image.substitute_values(img[:,:,0], mask, trueValueFin = 200, falseValueFin = img2[:,:,0], show = True)
# trueValueFin and falseValueFin are 3d images while img is 1d
image_lib.filter_image.substitute_values(img[:,:,0], mask, trueValueFin = img, falseValueFin = img2, show = True)
#trueValueFin and falseValueFin are 1d colours and img is 1d
image_lib.filter_image.substitute_values(img[:,:,0], mask, trueValueFin = 200, falseValueFin = 38, show = True)
# trueValueFin and falseValueFin are 3d colours while img is 1d
image_lib.filter_image.substitute_values(img[:,:,0], mask, trueValueFin = [200, 0, 0], falseValueFin = [0, 200, 0], show = True)

# trueValueFin is 1d colour and falseValueFin is 1d image and img is 3d -> does not work, use mask instead of img
image_lib.filter_image.substitute_values(mask, mask, trueValueFin = 200, falseValueFin = img2[:,:,0], show = True)
# image_lib.filter_image.substitute_values(img, mask, trueValueFin = 200, falseValueFin = img2[:,:,0], show = True)


plt.show()

    
