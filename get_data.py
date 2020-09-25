import tensorflow as tf
import os


# Download annotation files
annotation_folder = 'data/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'https://ivc.ischool.utexas.edu/VizWiz_final/caption/annotations.zip',
                                          extract = True)
    annotation_train = os.path.dirname(annotation_zip)+'/annotations/train.json'
    annotation_val = os.path.dirname(annotation_zip)+'/annotations/val.json'  
    annotation_test = os.path.dirname(annotation_zip)+'/annotations/test.json'

    os.remove(annotation_zip)


# Download image files
image_folder = 'data/Images/train/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip',
                                      extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = os.path.abspath('.') + image_folder

image_folder = 'data/Images/val/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('val.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip',
                                      extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = os.path.abspath('.') + image_folder

image_folder = 'data/Images/test/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('test.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin = 'https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip',
                                      extract = True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = os.path.abspath('.') + image_folder







