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