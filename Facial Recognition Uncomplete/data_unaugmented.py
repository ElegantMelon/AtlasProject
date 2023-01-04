from pickle import NONE
import tensorflow as tf
import os
import cv2
import json 
import numpy as np
from matplotlib import pyplot as plt
import splitfolders
#if no new data set block all this out

images = tf.data.Dataset.list_files('data\\images\\*.jpg')

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
images.as_numpy_iterator().next()


image_generator = images.batch(4).as_numpy_iterator()

plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)

#stop block here
#write code for auto split of data to test train and validate
#input_folder = 'data\\images\\'
#splitfolders.ratio(input_folder, output = "data", seed =42, ratio=(0.15, 0.7, 0.15), group_prefix=NONE)


for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data', folder,'images')):
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data', folder, 'labels', filename)
            os.replace(existing_filepath, new_filepath) 
            
    




