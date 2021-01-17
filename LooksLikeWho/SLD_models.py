'''
Module to train models and make predictions.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
'''

# IMPORTS

import os
import gc
import re
import csv
import cv2
import sys
import PIL
import math
import time
import uuid
import mtcnn
import scipy
import scipy.sparse
import random
import pathlib
import pickle
import pandas as pd
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

retval = os.getcwd()
print("[>>SLD MODEL<<] Current working directory %s" % retval, \
      "__NAME__ = ", __name__)

from LooksLikeWho.SLD_tools import *
from LooksLikeWho.SLD_quad_model import *

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from tensorflow import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, \
    Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Activation, Input, \
    Concatenate, multiply
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback as CB
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, \
    CSVLogger,EarlyStopping
from tensorflow.keras.utils import to_categorical  #EDITSLD
from tensorflow.keras import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# MODELS CATALOG
models_catalog=[
    'MobileNetV2', 
    'ResNet50', 
    'VGG16', 
    'InceptionV3', 
    'Xception', 
    'FaceNet']

models_objects=[
    MobileNetV2, 
    ResNet50, 
    VGG16, 
    InceptionV3, 
    Xception]

models_weight_paths=[
    r".\models\weights\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
    r".\models\weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
    r".\models\weights\facenet_keras_weights.h5"
    ]


# FUNCTIONS
def get_weight_path(model_name):
    '''
    Return the path where model weights are stored.

    Arguments:
        model_name: name of the model used

    Output:
        full path containing the model weights
        '''

    i=models_catalog.index(model_name)
    path=models_weight_paths[i]
    if not os.path.exists(path):
        path="."+path
    
    return path


def get_base_model(model_name, from_notebook=True):
    '''
    Create the base model object used for transfer learning.

    Arguments:
        model_name: string corresponding to the model
        weight_path: location of the base model weights

    Outputs:
        base model with preset weights
    '''
    if from_notebook:
        relative = ".."
    else:
        relative = "."
        
    weight_path = get_weight_path(model_name)
    # print('* Loading model weights from: '+path+'...')
    if model_name!="FaceNet":
        model = models_objects[models_catalog.index(model_name)]
    else:
        model_path = os.path.join(relative, "models", "weights", "facenet_keras.h5")
        print(model_path)
        model = load_model(model_path)
    
    # print('Base model input shape', IMAGE_WIDTH, IMAGE_HEIGHT)
    if model_name!="FaceNet":
        base_model = model(
        include_top=False,
        weights=weight_path,
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        )
    else:
        base_model = model
        
    return base_model


def train_quad_siamese(data_dir, test=False, from_notebook=False, verbose=True, \
                       realistic_method="average"):
    '''
    Train model and save weights.

    Arguments:
        data_dir: location of the train and test folders
        test: if test, only use 1 epoch and save results in a special directory
        from_notebook: use to update relative paths
        verbose: print progress and step completions
        realistic_method: "max" or "average". Decide how the predicted class 
            will be computed, by either selecting the class corresponding to 
            the image with the smallest distance ("min"), or by selecting the 
            class whose top 3 matches have the smallest average ("average").

    External Arguments:
        IMAGE_WIDTH: image width (used for input in base model)
        IMAGE_HEIGHT: image height (used for input in base model)
        BATCH_SIZE: batch size used for training
        MODEL_BASE_TAG: name of the base model (used to retrieve weights and 
            generate bottleneck features)
        EPOCHS: number of epochs

    Outputs:
        checkpoints: folder containing best model weights
        history.csv: history of model training
        history.png: plots of accuracy, loss, and learning rates
        summary.txt: model structure
    '''

    ## start timer for runtime
    time_start = time.time()
    print('\n\n******************* {} *******************\n'.format(MODEL_BASE_TAG))
    ## create location for train and test directory
    if not os.path.exists(data_dir):
        raise Exception("specified data directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'train')):
        raise Exception("training directory does not exist.")
    if not os.path.exists(os.path.join(data_dir,'test')):
        raise Exception("specified test directory does not exist.")

    # Get the specific name of the dataset
    dataset_tag = data_dir.split('\\')[-1]

    ## adjust relative path
    if from_notebook:
        base_weight_path = '.' + get_weight_path(MODEL_BASE_TAG)
        relative = ".."
    else:
        base_weight_path = get_weight_path(MODEL_BASE_TAG)
        relative = ".."
    
    populate_classes(data_dir, MODEL_BASE_TAG, dataset_tag)
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    ## initialize datagenerator
    datagen = ImageDataGenerator(rescale=1/255.)

    ## run generator
    if verbose: print("* Creating generators...")
    
    # Note: one subdirectory per class in the train/test folder.
    print('   > ', end='')
    generator_train = datagen.flow_from_directory(
        train_dir,
        color_mode="rgb",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
        )

    print('   > ', end='')
    generator_test = datagen.flow_from_directory(
        test_dir,
        color_mode='rgb',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
        )
    
    # extract info from generator
    ## extract info from generator
    
    train_filenames=generator_train.filenames
    test_filenames=generator_test.filenames
    
    # Save train filename for identification use later on
    np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_filenames.npy'), train_filenames)
    nb_train_samples = len(train_filenames)
    nb_test_samples = len(test_filenames)
    # print(generator_test.filenames) # ['n000001\\0001_01.jpg', 'n000001\\0002_01.jpg', 'n000001\\0003_01.jpg',
    num_classes = len(generator_train.class_indices)
    num_step_train = int(math.ceil(nb_train_samples / BATCH_SIZE))  
    num_step_test = int(math.ceil(nb_test_samples / BATCH_SIZE))  
    # num_classes = generator_train.classes_count

    ## create path for models if needed
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))

    ## check if bottleneck weights exist   
    if not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy')):
        if verbose: print("* Creating bottleneck features...")
        base_model = get_base_model(MODEL_BASE_TAG)

        ## create bottle neck by passing the training data into the base model
        print('   > ', end='')
        bottleneck_features_train = base_model.predict(
                                                    generator_train,
                                                    steps=num_step_train,
                                                    verbose=1
                                                    )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'), bottleneck_features_train)
        del bottleneck_features_train

        ## create bottle neck by passing the training data into the base model
        print('   > ', end='')
        bottleneck_features_test = base_model.predict(
                                                    generator_test,
                                                    steps=num_step_test,
                                                    verbose=1
                                                    )
        ## save bottleneck weights
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_test.npy'), bottleneck_features_test)
        del bottleneck_features_test
        #_gc.collect()
        
    ## load the bottleneck features saved earlier  
    if verbose: print("* Loading bottleneck features...")
    # print(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'))
    #_gc.collect()
    train_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_train.npy'))
    #_gc.collect()
    test_data = np.load(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_bottleneck_features_test.npy'))


    ## get the class labels for the training data, in the original order  
    train_labels = generator_train.classes
    test_labels = generator_test.classes
    
    print("Train_data shape: {}, train_labels shape: {}.".format(train_data.shape, train_labels.shape))
    
    ## Shuffle train and test set - NOT USED
    # np.random.seed(1986)
    orders = np.arange(train_data.shape[0])
    # np.random.shuffle(orders)
    # train_data = train_data[orders]
    # train_labels = train_labels[orders]
    ORDERS=orders

    
    ## Save ORDERS for predictions later
    np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_orders.npy'), ORDERS)

    if verbose: print("   > Training data shape", train_data.shape)
    if verbose: print("   > Testing data shape", test_data.shape)
    if verbose: print("   > Training label data shape", train_labels.shape)
    if verbose: print("   > Testing label data shape", test_labels.shape)

    ## convert the training labels to categorical vectors
    if verbose: print("* Encoding labels...")
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    # Save classes
    if verbose: print("* Saving classes as sparse matrices...")
    
    if not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_sparse.npz')) \
        or not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_test_labels_sparse.npz')) \
            or not os.path.exists(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_AWS.npz')):
            
        scipy.sparse.save_npz(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_sparse.npz'), \
                          scipy.sparse.csc_matrix(train_labels))
        
        scipy.sparse.save_npz(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_test_labels_sparse.npz'), \
                          scipy.sparse.csc_matrix(test_labels))
            
        # Create a smaller train_label matrix for fast loading in AWS.
        train_labels_AWS = np.argmax(train_labels, axis=1)
                
        np.save(os.path.join(relative, 'models', 'bottlenecks', MODEL_BASE_TAG + dataset_tag + '_train_labels_AWS.npy'), \
                  train_labels_AWS)
    
    ## build other CNNs
    if verbose: print("* Creating model...")
    
    network = build_network(input_shape=train_data.shape[1:], embeddingsize=EMBEDDINGSIZE)
    metricnetwork = build_metric_network(single_embedding_shape=[EMBEDDINGSIZE])
    model = build_quad_model(train_data.shape[1:], network, metricnetwork, margin=MARGIN, margin2=MARGIN2)

    ## save info about models
    # model_name = re.sub("\.","_",str(MODEL_VERSION))
    model_name = MODEL_VERSION

    ## create directory for version specific
    if not os.path.exists(os.path.join(relative,"models")):
        os.mkdir(os.path.join(relative,"models"))
    if not os.path.exists(os.path.join(relative,"models","runs",model_name)):
        os.mkdir(os.path.join(relative,"models","runs",model_name))
    if not os.path.exists(os.path.join(relative,"models","runs",model_name,"checkpoints")):
        os.mkdir(os.path.join(relative,"models","runs",model_name,"checkpoints"))

    ## OPTIMIZER
    #optimizer = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999)
    optimizer = Adam(lr=START_LR)
    # optimizer = Adam(lr=0.005)

    ## compile model
    model.compile(
        optimizer=optimizer,
        loss=None
        # metrics=["accuracy"]
        )

    # Compute generators
    gentrain = generate_quad(train_data, train_labels, batch_size=BATCH_SIZE)
    # gentest = generate_quad(test_data, test_labels, batch_size=BATCH_SIZE)
    gentest=0 # genetest is deprecated.
    
    ## CALLBACKS
    ## progress
    if from_notebook:
        callbacks = []
        model_verbose = 2
    else:
        callbacks = []
        model_verbose = 2

    N, k = N_ONESHOTMETRICS, k_ONESHOTMETRICS #N_ONESHOTMETRICS not used at the moment
    callbacks.append(OneShotMetricsQuad(network, metricnetwork, N, k, gentest, \
        test_data, test_labels, train_data, train_labels, realistic_method=realistic_method))

    ## reduce LR
    lrate = ReduceLROnPlateau(
        monitor="loss",
        factor = 0.4,
        patience=1,
        verbose=1,
        min_lr = 0.0000001
    )
    callbacks.append(lrate)

    ## early stopping
    es = EarlyStopping(
        monitor='loss',
        mode='min',
        verbose=1,
        patience=10,
        min_delta=0.005
        )
    callbacks.append(es)

    ## save
    checkpoints = ModelCheckpoint(
        os.path.join(relative,"models","runs",model_name,"checkpoints",model_name+".h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        # period=1
        )
    # callbacks.append(checkpoints)

    ## compute step size and epoch count
    n_epochs = EPOCHS

    ## adjust parameters for test
    if test:
        n_epochs = 1

    ## save model summary
    if verbose: print("* Saving model summary...")
    with open(os.path.join(relative,"models","runs",model_name,"summary"+CUSTOM_FILE_NAME+".txt"),'w') as fh:
        # pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    if verbose: print("* Training model...")
    history = model.fit(gentrain,
                        # train_data,
                        # train_labels,
                        epochs=n_epochs,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        # batch_size=BATCH_SIZE, 
                        verbose=1,
                        # validation_data=(test_data, test_labels),
                        callbacks=callbacks,
                        # class_weight=dict(enumerate(class_weights))
                        )

    ## save model
    if verbose: print("* Saving main model...")
    model.save(os.path.join(relative,"models","runs",model_name,'my_model_weights.h5'))
    
    if verbose: print("* Saving encoder model...")
    network.save(os.path.join(relative,"models","runs",model_name,'encoder_model_weights.h5'))
    
    if verbose: print("* Saving similarity model...")
    metricnetwork.save(os.path.join(relative,"models","runs",model_name,'similarity_model_weights.h5'))
    
    ## save history into a csv file
    if verbose: print("* Saving training history...")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(relative,"models","runs",model_name,"history.csv"))

    ## create plots for learning rate, accuracy, and loss
    save_path = os.path.join(relative,"models","runs",model_name)
    plot_history_quad(history_df, save_path, custom_file_name=CUSTOM_FILE_NAME) 
    
    ## compute running time
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## All finished
    if verbose: print("* Done.")
    
    
def load_models_and_data(model_name, dataset_tag, from_notebook=True, online=False):
    '''    
    Load models (base, encoder and similarity), the train data and the labels,
        the map between the labels and the actuals names and the image input size.

    Arguments:
        model_name: string (FaceNet, ResNet50, InceptionV3, MobileNetV2, Xception, VGG16).
        dataset_tag: string to represent the dataset used. 
        from_notebook: use True to offset relative paths.
        online: use True when deployed on AWS.
        
    Outputs:
        base_model: the pre-encoder model.
        network: the encoder model.
        metricnetwork: the similarity function.
        data_path: the path to the train and test set.
        train_data: an np.array containing the train set data
        train_labels_AWS: an np.array containg the train labels, optimized for AWS.
        train_filenames: the filenames corresponding to the train data.
        classes: the names of all the classes.
        target_size: the image size required for the pre-encoder. 
    '''
    
    # Adjust relative path
    if from_notebook:
        relative = ".."
    else:
        relative = "."
    
    if not online:
        data_path=os.path.join(r"C:\DATASETS\1. VGGFace2", dataset_tag)
    else:
        data_path=os.path.join(r".\app\static\datasets", dataset_tag)
    
    # Get path and load base model
    try:
        base_model = get_base_model(model_name, from_notebook)
    except:
        raise Exception("No base model weights could be located")

    # Get paths and load encoder and similarity models
    network_path=os.path.join(relative, "models", "runs", model_name + dataset_tag, "encoder_model_weights.h5")
    metricnetwork_path=os.path.join(relative, "models", "runs", model_name + dataset_tag, "similarity_model_weights.h5")
    
    ## Check if models exists
    if not os.path.isfile(network_path):
        raise Exception("No encoder model in specified directory {}".format(network_path))

    if not os.path.isfile(metricnetwork_path):
        raise Exception("No encoder model in specified directory {}".format(metricnetwork_path))
    
    ## Load models
    network=load_model(network_path)
    metricnetwork=load_model(metricnetwork_path)
    
    # Load train data and labels
    try:
        print(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag + '_bottleneck_features_train.npy'))
        print(os.path.join(relative, 'models', 'bottlenecks', model_name + dataset_tag + '_train_labels_AWS.npy'))
        gc.collect()
        train_data = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag + '_bottleneck_features_train.npy'))
        gc.collect()
        # train_labels = np.array(scipy.sparse.load_npz(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag + '_train_labels_sparse.npz')).todense())
        train_labels_AWS = np.load(os.path.join(relative, 'models', 'bottlenecks', model_name + dataset_tag + '_train_labels_AWS.npy'))
    except:
        raise Exception('Train data or labels cannot be retrieved from given folders.')
        
    # Load mapping between the label columns and the label names
    # orders = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag +  '_orders.npy'))
    
    # Set the target image size for the base model
    target_size=(160,160) if model_name == 'FaceNet' else (224,224)
    
    # Load train set filenames
    train_filenames=np.load(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag + '_train_filenames.npy'))
    
    # Load classes
    classes = np.load(os.path.join(relative, "models", 'bottlenecks', model_name + dataset_tag + '_CLASSES.npy'))
        
    return base_model, network, metricnetwork, data_path, train_data, train_labels_AWS, train_filenames, classes, target_size


def make_prediction_quad(model_name, base_model, network, metricnetwork, image_path, \
                        train_data, train_labels, train_filenames, classes, data_path, \
                        target_size=(224,224), filename=None, method="average", \
                        extra_matches=None, online=False):
    '''
    Arguments:
        model_name: name of the model, e.g. "FaceNet"
        base_model: the pre-encoder model.
        network: the encoder model.
        metricnetwork: the similarity function.
        image_path: path of the image to be tested.
        train_data: an np.array containing the train set data
        train_labels_AWS: an np.array containg the train labels, optimized for AWS.
        train_filenames: the filenames corresponding to the train data.
        classes: the names of all the classes.
        data_path: path to train and test sets.
        target_size: the image size required for the pre-encoder.
        filename: filename of the tested image.
        method: "max" or "average". Decide how the predicted class will be computed,
            by either selecting the class corresponding to the image with the smallest
            distance ("min"), or by selectong the class whose top 3 matches have the
            smallest average ("average").
        extra_matches: number of extra matches to return (by order of probability).
        online: use rue when deployed on AWS.

    Outputs:
        pred_class: predicted class of the tested image.
        distance: similarity between the tested image and the matched image.
        actual_image_index: index of thee match image in the dataset.
        actual_image_path: path of the match image.
        matches_list: list of Match objects.
    '''

    # Center/cropp image using MTCNN
    print("* Starting face detection...")
    detector = MTCNN()
    img = img = cv2.imread(image_path)
    full_height, full_width, _ = img.shape
    
    detections = detector.detect_faces(img)
    print(detections)
    x1, y1, width, height = detections[0]['box']
    # Make image square
    w2=width//2
    xc = x1+w2
    h2=height//2
    yc = y1+h2

    d=max(height,width)//2
    print(yc-d,yc+d, xc-d,xc+d)
    Y0, Y1, X0, X1 = yc-d, yc+d, xc-d, xc+d
    
    # Check that nothing is outside the frame
    check = all([
        Y0>=0,
        X0>=0,
        Y1<=full_height,
        X1<=full_width,
        ])
    
    if not check:
        print("FACE PARTIALLY SHOWN ON ORIGINAL IMAGE (TOO ZOOMED IN): making the image square as-is.")
        dmin=min(yc, full_height-yc, xc, full_width-xc)
        Y0, Y1, X0, X1 = yc-dmin, yc+dmin, xc-dmin, xc+dmin
        print("New coords:", Y0, Y1, X0, X1)
        
    face = img[Y0:Y1, X0:X1, ::-1]

    # resize pixels to the model size
    face = PIL.Image.fromarray(face, mode='RGB')
    # if MODEL_BASE_TAG == 'FaceNet':
    #     target_size=(160,160)
    face = face.resize(target_size, Image.BICUBIC)
    face = np.asarray(face)
    
    display_image(face)
    image = np.asarray(face)
    
    # pre-process
    image = image / 255
    image = np.expand_dims(image, axis=0)
    
    # Get the embedding using the base model
    image=base_model.predict(image)
    
    pred_class, distance, actual_image_index, sorted_predicted_cats, \
        sorted_distances, sorted_actual_image_index = \
        compute_learned_dist_one_vs_all(network, metricnetwork, 1, train_data, \
            train_labels, test_data=None, test_labels=None, output_labels=1, \
            also_get_loss=0, verbose = 1, label="realistic", method=method, \
            predict=1, predict_model_name=model_name, image=image)   

    print("[xxxx] Current working directory %s" % retval, \
      "__NAME__ = ", __name__)
    actual_image_path=os.path.normpath(train_filenames[actual_image_index])
    print("actual_image_path", actual_image_path)
    full_actual_image_path=os.path.join(data_path, "train", actual_image_path).replace('\\','/')
    print("full_actual_image_path", full_actual_image_path)
    print("OPENING PREDICTED IMAGE")
    actual_image=Image.open(full_actual_image_path)
    
    if not online:
        display_image(actual_image)

    pred_class=int(pred_class)
    pred_name=classes[pred_class].replace("_"," ").replace('"', '')
    print("Predicted class:", pred_class, "- corresponding to:", pred_name)
    print("Distance:", distance)
    print("Path:", full_actual_image_path)
    
    # Save image as file if filenmae is provided
    if filename:
        if not online:
            plt.savefig(os.path.join(".", "app", "static", "prediction", "prediction_"+filename+".png"), dpi=150)
            # plt.savefig("./app/static/prediction/prediction_{}.png".format(filename), dpi=150)
        else:
            # actual_image.save("./app/static/prediction/prediction_{}.png".format(filename))
            actual_image.save(os.path.join(".", "app", "static", "prediction", "prediction_"+filename+".png"))
        
    if extra_matches:
        matches_list=[]
        for i in range(extra_matches+1): # We already have the first match
            if i<=len(sorted_actual_image_index):
                path=os.path.normpath(train_filenames[sorted_actual_image_index[i]].replace('\\',"____")) # four underscore, will be split later in views:: display_train_sample
                print("normalized path", path)
                # path=retrieve_train_image_path(train_filenames[sorted_actual_image_index[i]], data_path)
                # convert path to html path
                # path=pathlib.Path(path).as_uri()
                
                name=classes[sorted_predicted_cats[i]].replace("_"," ").replace('"', '')
                distance=sorted_distances[i]
                matches_list.append(Match(path, name, distance))
                # print("****")
                # print(path, name, distance)
                # image=Image.open(path)
                # display_image(image)
                
    if extra_matches:
        return  pred_class, distance, actual_image_index, actual_image_path, matches_list
    else:
        return  pred_class, distance, actual_image_index, actual_image_path
    
# END OF CODE