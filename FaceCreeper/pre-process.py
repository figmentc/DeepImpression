
from random import shuffle
# from PIL import Image, ImageEnhance
from scipy.misc import imread
from scipy.misc import *
from numpy import *
import sys 
import os 


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.

filepath = sys.argv[1]
os.rename(filepath, filepath+".png")
im = array(imread(filepath+".png"))
# print type(im)
cropDim = map(int, map(float, sys.argv[2].split(',')))
# print cropDim
# print im
# print im.shape
# im = imresize(im, [32,32])
im = rgb2gray(im)
im = rot90(im, 3)
im = im[cropDim[0]:cropDim[2], cropDim[1]:cropDim[3]]
im = imresize(im, [32,32])


imsave(filepath+".png", im);
print filepath+'.png'





# def get_all_images(path):
# 	sep = '/'
# 	all_pics = array([])
#     for filename in os.listdir(path):
#         img = imread(path + sep + filename)
#         all_pics.append(img)
#     return all_pics

# # modify these fucking constants to your damn preferences
# PATH = "AngieHarmon"
# PERCENT_CONTRAST = 	10
# PERCENT_MAX_SHARP =	10
# PERCENT_MIN_SHARP = 10
# PERCENT_BRIGHT = 	10


# # example i/o: percent_to_value(700,5) --> 35
# def percent_to_value(list_size, percent):
# 	return (list_size * 5) / 100


# # returns list of enhanced images
# def main():
# 	#variables
# 	image_list = get_all_images(PATH)
# 	list_size = len(image_list)
# 	counter = 0
# 	new_list = []

# 	# converts all percent to value
# 	value_contrast = percent_to_value(list_size, PERCENT_CONTRAST)
# 	value_max_sharp = percent_to_value(list_size, PERCENT_MAX_SHARP) + value_contrast
# 	value_min_sharp = percent_to_value(list_size, PERCENT_MIN_SHARP) + value_max_sharp
# 	value_bright = percent_to_value(list_size, PERCENT_BRIGHT) + value_min_sharp

# 	# shuffles images
# 	shuffle(image_list)

# 	# loops through all images, modifies them
# 	for image in image_list:
# 		# enhance all to B/W first
# 		# An enhancement factor of 0.0 gives B/W image; 1.0 gives original image
# 		image = ImageEnhance.Color(image)
# 		image.enhance(0.0)

# 		if counter < value_contrast:
# 			# An enhancement factor of 0.0 gives a solid grey image; 1.0 gives original image
# 			contrast = ImageEnhance.Contrast(image)
# 			new_list.append(contrast.enhance(2))
		
# 		elif counter < value_max_sharp:
# 			# Maximum sharpness
# 			maxSharp = ImageEnhance.Sharpness(image)
# 			new_list.append(maxSharp.enhance(2))
		
# 		elif counter < value_min_sharp:
# 			# minimum sharpness
# 			minSharp = ImageEnhance.Sharpness(image)
# 			new_list.append(minSharp.enhance(0))
		
# 		elif: counter < value_bright
# 			# An enhancement factor of 0.0 gives a black image; 1.0 gives original image
# 			brightness = ImageEnhance.Brightness(image)
# 			new_list.append(brightness.enhance(0.5))
		
# 		else:
# 			# append rest of images to list
# 			new_list.append(image)
		
# 		counter++

# 	return new_list