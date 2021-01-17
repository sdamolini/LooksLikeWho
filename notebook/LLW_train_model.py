# -*- coding: utf-8 -*-
"""
Code to create datasets and train the models.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
"""

# IMPORTS

import os
import gc
import sys
sys.path.append('..')
import imp
import numpy as np
import LooksLikeWho
from LooksLikeWho.SLD_tools import *
from LooksLikeWho import SLD_models
from LooksLikeWho.SLD_models import *


############################### USER INPUTS ##################################

# PARAMETERS AND SETTING FOR THE DATASET CREATION AND TRAINING

## Technical parameters
local=True            # True if running locally
memory_issues=False   # True if GPU memory issues ("failed to allocate" / OOM)
disable_gpu=False     # True if above setting is not sufficient

## Generate train/test set
generate_train_test_sets=False  # True to gen train/test set, run only once

## Select data set (select only one amongst choices)
dataset_tag = "_litecropped" # 2 x 10 x 4
# dataset_tag = "_medium_cropped" # 2 x 100 x 4
# dataset_tag = "_mediumcroppedx10"# 2 x 100 x 10
# dataset_tag = "_ALL-HQ-UNZOOMED" 2 x 9131 x 4
# dataset_tag = "_ALL-HQ-UNZOOMED-10X" # 10931 x 10 (train) + 1931 x 4 (test)

## Set training parameters
SLD_models.MODEL_BASE_TAG = 'FaceNet'      # Select model among:
    # 'MobileNetV2', 'ResNet50', 'VGG16', 'InceptionV3', 'Xception', 'FaceNet'
SLD_models.CUSTOM_FILE_NAME= "_quad_final" # Custom note
SLD_models.BATCH_SIZE = 8                  # Use 8 for final run
SLD_models.EPOCHS = 2                      # Use 6 for final run
SLD_models.STEPS_PER_EPOCH = 20          # Use 2000 for final run
SLD_models.k_ONESHOTMETRICS = 10           # Use 10 final run
SLD_models.START_LR = 0.001                # Adam default 0.001
SLD_models.MARGIN = 0.25                   # Use 0.25 for final run
SLD_models.MARGIN2 = 0.03                  # Use 0.03 for final run
SLD_models.EMBEDDINGSIZE = 128             # Use 10 final run

########################## END OF USER INPUTS ################################


# SETTINGS FOR PROPER FILE PATHS
if local:
    data_path=os.path.join(r"C:\DATASETS\1. VGGFace2", dataset_tag)
else:
    data_path=os.path.join(r".\datasets", dataset_tag)


# SPECIAL SETTING IF GPU MEMORY ISSUES
if memory_issues:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
          print("___OK___")
      except RuntimeError as e:
        print(e)

if disable_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# CREATE SUB DATASET FROM VGGFACE2 - RUN ONLY ONCE
if generate_train_test_sets:
    ## Get train and test set with face cropping
    source=r"D:\DATASETS\VGGFACE2_FULL"
    dest=data_path
    ## Generate train set
    get_what_from_full_set_with_face_crop("train", source, dest, \
        max_samples_per_class=10, max_classes=9131, mini_res=160)
    ## Generate test set
    get_what_from_full_set_with_face_crop("test", source, dest, \
        max_samples_per_class=4, max_classes=9131, mini_res=160)


# PARAMETERS AUTOMATICALLY SET UP
SLD_models.N_ONESHOTMETRICS = 3 # Parameter not used right now
SLD_models.IMAGE_WIDTH, SLD_models.IMAGE_HEIGHT = \
    (160, 160) if SLD_models.MODEL_BASE_TAG=="FaceNet" else \
        (224,224)
SLD_models.CUSTOM_FILE_NAME+= \
    "_B"+str(SLD_models.BATCH_SIZE)+ \
    "_E"+str(SLD_models.EPOCHS)+ \
    "_S"+str(SLD_models.STEPS_PER_EPOCH)+ \
    "_k"+str(SLD_models.k_ONESHOTMETRICS)+ \
    "_lr"+str(SLD_models.START_LR)+ \
    "_M"+str(SLD_models.MARGIN)+ \
    "_MM"+ str(SLD_models.MARGIN2)+ \
    "_em"+str(SLD_models.EMBEDDINGSIZE)
SLD_models.MODEL_VERSION = \
    SLD_models.MODEL_BASE_TAG+dataset_tag 
                    
    
### TRAIN MODEL ###
gc.collect()
np.set_printoptions(suppress=True)
##############################################################################
train_quad_siamese(data_path, test=False, from_notebook=True, \
    verbose=True, realistic_method="average")
##############################################################################

# END OF CODE