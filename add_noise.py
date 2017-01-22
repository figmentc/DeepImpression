import sys
import numpy as np
from pylab import savefig
import matplotlib.pyplot as plt
import random, os
from os.path import sep
from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle
import cPickle
from PIL import Image, ImageEnhance

##########################################################################
## This file adds some variety to the training data.
##########################################################################

## contrast: An enhancement factor of 0.0 gives a solid grey image.
## A factor of 1.0 gives the original image.
#contrast = ImageEnhance.Contrast(image)
#contrast.enhance(2).show()

## Maximum sharpness
#sharpeness = ImageEnhance.Sharpness(image)
#sharpeness.enhance(2).show()

## An enhancement factor of 0.0 gives a black image.
## A factor of 1.0 gives the original image.
#brightness = ImageEnhance.Brightness(image)
#brightness.enhance(0.5).show()

## An enhancement factor of 0.0 gives a black image.
## A factor of 1.0 gives the original image.
#darken = ImageEnhance.Brightness(image)
#darken.enhance(1.5).show()

def add_noise(data, sharpeness, brightness, darkness, contrast):
    """
        Choose what percentage of each feature you want to see in 
        the dataset.
        data is a list of images.
        sharpeness, brightness, darken, contrast are percentages.
        Return alterred data set.
    """
    len_data = len(data)
    num_sharpness = len_data * sharpeness
    num_brightness = len_data * brightness
    num_darkness = len_data * darkness
    num_contrast = len_data * contrast

    data = random.shuffle(data)

    sharpen = data[0 : num_sharpness]
    brighten = data[num_sharpness : num_sharpness + num_brightness]
    darken = data[num_sharpness + num_brightness : num_sharpness + num_brightness + num_contrast]
    contrast = data[num_sharpness + num_brightness + num_contrast : ]
    
    for sharp_thing in sharpen:
        sharpeness = ImageEnhance.Sharpness(image)
        sharpeness.enhance(2)
    for contrast_thing in contrast:
        contrast = ImageEnhance.Contrast(image)
        contrast.enhance(2)
    for bright_thing in brighten:
        brightness = ImageEnhance.Brightness(image)
        brightness.enhance(0.5)
    for dark_thing in darken:
        darken = ImageEnhance.Brightness(image)
        darken.enhance(1.5)

    big_list = sharpen + brighten + darken + contrast

    return big_list






