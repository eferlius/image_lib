import cv2

def n_channels(img):
    if len(img.shape) == 3:
        return img.shape[2]
    else:
        return 1

def colour_according_to_img(img):
    if len(img.shape) == 3:
        return [0]*img.shape[2]
    elif len(img.shape) == 2:
        return 0
    
def scalar_in_range(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def from_list_to_list_of_int(my_list):
    return list(map(int, my_list))

def from_list_of_list_to_list_of_list_of_int(my_list):
    return [[int(num) for num in sub] for sub in my_list]

def convert_to_grayscale(input_image):
    if len(input_image.shape) == 2:
        return input_image
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image