# -*- coding: utf-8 -*-
"""
Code to test the functions that are going to be deployed on AWS.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
"""

# IMPORTS

import os, gc, sys
sys.path.append('..')
import LooksLikeWho
from PIL import Image
import matplotlib.pyplot as plt


############################### USER INPUTS ##################################

# PARAMETERS AND SETTING FOR TESTING THE MODEL WITH A TEST IMAGE

# Select dataset (uncomment one)
# dataset_tag = "_litecropped" # 2 x 10 x 4
# dataset_tag = "_medium_cropped" # 2 x 100 x 4
# dataset_tag = "_mediumcroppedx10"# 2 x 100 x 10
# dataset_tag = "_ALL-HQ-UNZOOMED" 2 x 1931 x 4
dataset_tag = "_ALL-HQ-UNZOOMED-10X" # 10931 x 10 (train) + 1931 x 4 (test)

# Set parameters
method="min"
model_name="FaceNet"
from_notebook=True # <<<-------------  TO CHANGE FOR AWS
online=False if __name__ == "__main__" else True
object_sizes=False        # Output the size of preloaded objects

# Chose a test image below (uncomment one)

# img_test = "sm.jpg"      # sophie marceau
# img_test = "da.jpg"      # david hasselhoff
# img_test = "eg.jpg"      # elodie gaussuin
# img_test = "yn.jpg"      # yannick noah
# img_test = "ab.jpg"      # a bouteflika (from train set!)
img_test = "ines.jpg"    # ines de la fressange
# img_test = "aff.jpg"     # a fine Fenzy from _litecropped (train set!)
# img_test = "ar.jpg"      # a raja _litecropped (train set!)

########################## END OF USER INPUTS ################################


# LOAD EVERYTHING NEEDED TO RUN THE TEST
base_model, \
network, \
metricnetwork, \
data_path, \
train_data, \
train_labels_AWS,\
train_filenames, \
classes, \
target_size = LooksLikeWho.SLD_models.load_models_and_data(model_name, \
              dataset_tag, from_notebook, online)


# OUTPUT SIZE OF OBJECTS FOR OPTIMIZATION
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

if object_sizes:
    for o in [base_model, network, metricnetwork, data_path, train_data, \
              train_labels_AWS ,train_filenames, classes, target_size]:
        print(namestr(o, globals()))
        print(round(sys.getsizeof(o)/1e6,0), "MB")


# DISPLAY INPUT IMAGE --> MAKE SURE TO TYPE %MATPLOTLIB INLINE FIRST
img_test_path=os.path.join(r"..\app", "static", "examples", img_test)
image= Image.open(img_test_path)
plt.imshow(image)


# RUN FOLLOWING CODE IF GPU MEMORY ISSUES
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# PREDICT
gc.collect()
##############################################################################
pred_class, distance, actual_image_index, actual_image_path, matches_list = \
    LooksLikeWho.SLD_models.make_prediction_quad(model_name, base_model, \
    network, metricnetwork, img_test_path, train_data, train_labels_AWS, \
    train_filenames, classes, data_path, target_size, \
    method="average", extra_matches=5)
##############################################################################

# END OF CODE