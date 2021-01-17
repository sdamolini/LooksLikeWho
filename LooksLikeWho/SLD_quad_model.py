# -*- coding: utf-8 -*-
'''
Contains the main classes and functions to set up a quadruplet loss face
identification CNN.

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
import joblib
import random
import pickle
import pandas as pd
import numpy as np
import numpy.random as rng
from mtcnn.mtcnn import MTCNN
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from LooksLikeWho.SLD_tools import *
import LooksLikeWho.SLD_models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from tensorflow import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, Activation, Input, Concatenate, multiply
from tensorflow.keras.layers import Lambda, Layer
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
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger,EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras.losses import BinaryCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# CLASSES
class Match:
    '''
    A simple class to store matches from the algorithm prediction resulting 
    from a test image.
    Contains the name, path and distance.
    Data is used by the deployed app (in particular the carousel).
    
    Arguments:
        path: path of the image file.
        name: name of the image.
        distance: float, distance (similarity) betwwen the image and its match.
        
    Output:
        A Match object containing all arguments.
    '''
    
    
    def __init__(self, path=None, name=None, distance=None):
        self.path=path
        self.name=name
        self.distance=distance


class QuadrupletLossLayer(Layer):
    '''
    A custom tf.keras layer that computes the quadruplet loss from distances
    ap_dist, an_dist, and nn_dist.
    The computed loss is independant from the batch size.
    
    Arguments:
        alpha, beta: margin factors used in the loss formula.
        inputs: (ap_dist, an_dist, nn_dist) with:
            ap_dist: distance between the anchor image (A) and the positive 
                image (P) (of the same class),
            an_dist: distance between the anchor image (A) and the first image
                of a different class (N1),
            nn_dist: distance between the two images from different classes N1
                and N2 (that do not belong to the anchor class).

    External Arguments:
        LooksLikeWho.SLD_models.BATCH_SIZE: batch size used for training

    Output:
        The quadruplet loss per sample (averaged over one batch).
    '''
    
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
  
        super(QuadrupletLossLayer, self).__init__(**kwargs)
    
    def quadruplet_loss(self, inputs):
        ap_dist,an_dist,nn_dist = inputs
        
        #square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)
        
        return (K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) \
            +K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)) \
            /LooksLikeWho.SLD_models.BATCH_SIZE
    
    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return config
    
    
class OneShotMetricsQuad(CB):
    '''
    A custom callback to compute metrics that are very specific to siamese
    network. 
    
    Arguments:
        network: encoder network.
        metricnetwork: similarity function.
        N: (deprecated) Number of samples to use when testing an image,
            now all are used.
        k: number of tests to run per epoch, should be 1 if predict is 1
        gen: (deprecated) Generated test batches.
        train_data: train data (numpy array), the reference image will be 
            compared to all images contained in that array
        train_labels: train labels (numpy array)
        test_data: test data (numpy array), the reference image will be drawn
            amongst images contained in that array
        test_labels: test labels (numpy array)
        realistic_method: "max" or "average". Decide how the predicted class 
            will be computed, by either selecting the class corresponding to 
            the image with the smallest distance ("min"), or by selecting the 
            class whose top 3 matches have the smallest average ("average").
    
    Outputs:
        A 'train' metric is computed using 1 sample from the train set vs. all
        samples from the train set, with the expectations of a high accuracy.
        A 'realistic' metric is computed using 1 sample from the test set vs.
        all samples from the train set, with the expectations real-world 
        accuracy.
        All metrics are saved to 'logs'.
        Live metrics are also printed to the console during training. 
    ''' 

    def __init__(self, network, metricnetwork, N, k, gen, test_data, \
                 test_labels, train_data, train_labels, realistic_method):
        self.gen=gen
        self.test_data=test_data
        self.test_labels=test_labels
        self.train_data=train_data
        self.train_labels=train_labels
        self.k=k
        self.N=N
        self.metricnetwork=metricnetwork
        self.network=network
        self.realistic_method=realistic_method
    
    def on_train_begin(self, logs={}):
        # N-way one-shot learning accuracy
        self.one_shot_accuracy_train = []
        self.one_shot_accuracy_realistic = []
        self.one_shot_loss_train = []
        self.one_shot_loss_realistic = []
        self.one_shot_exact_matches_train = []
        self.one_shot_exact_matches_realistic = []

    def on_epoch_end(self, epoch, logs):
        time_start_epoch_eval = time.time()
        print(" ")
        #gc.collect()
        percent_correct_train, loss_train, exact_matches_train=\
            compute_learned_dist_one_vs_all(network=self.network, \
            metricnetwork=self.metricnetwork, k=self.k, train_data=self.train_data, \
            train_labels=self.train_labels, test_data=self.train_data , test_labels=self.train_labels, \
            also_get_loss=1, verbose = 1, label="train", method=self.realistic_method)

        percent_correct_realistic, loss_realistic, exact_matches_realistic=\
            compute_learned_dist_one_vs_all(network=self.network, \
            metricnetwork=self.metricnetwork, k=self.k, train_data=self.train_data, \
            train_labels=self.train_labels, test_data=self.test_data , test_labels=self.test_labels, \
            also_get_loss=1, verbose = 1, label="realistic", method=self.realistic_method)

        osa_train=percent_correct_train/100 # return a fraction and not a percentage
        osa_realistic=percent_correct_realistic/100 # return a fraction and not a percentage
        self.one_shot_accuracy_train.append(osa_train)
        self.one_shot_accuracy_realistic.append(osa_realistic)
        self.one_shot_loss_train.append(loss_train)
        self.one_shot_loss_realistic.append(loss_realistic)
        self.one_shot_exact_matches_train.append(exact_matches_train/100)
        self.one_shot_exact_matches_realistic.append(exact_matches_realistic/100)
        logs['one_shot_accuracy_train']=osa_train
        logs['one_shot_accuracy_realistic']=osa_realistic
        logs['one_shot_loss_train']=loss_train
        logs['one_shot_loss_realistic']=loss_realistic
        logs['one_shot_exact_matches_train']=exact_matches_train/100
        logs['one_shot_exact_matches_realistic']=exact_matches_realistic/100
        m, s = divmod(time.time()-time_start_epoch_eval, 60)
        h, m = divmod(m, 60)
        runtime = "%03d:%02d:%02d"%(h, m, s)
        print("Epoch evaluation runtime: ", runtime, "\n")
        print("*****SUMMARY*****:")
        print("Average of score for the train set:     {}%.    (100% is best, \
              0% is worst)".format(str(round(percent_correct_train,0))))
        print("Average of score for the realistic set: {}%.    (100% is best, \
              0% is worst)".format(str(round(percent_correct_realistic,0))))
        print('\n\n')
        gc.collect()
        return None
    
    
# FUNCTIONS

def build_network(input_shape, embeddingsize):
    '''
    Defines the neural network to refine image embeddings.
    
    Arguments: 
            input_shape: shape of input images.
            embeddingsize: vector size used to encode our picture.
            
    Outpout:
            A model to output refined embeddings from a pre-encoded image.
    '''

    # Convolutional Neural Network
    network = Sequential(name="encoder")
    network.add(Flatten(input_shape=input_shape))
    network.add(Dropout(0.30))
    network.add(Dense(embeddingsize, activation=None,
                    kernel_initializer='he_uniform'))
    
    return network


def build_metric_network(single_embedding_shape):
    '''
    Defines the neural network to learn the metric (similarity function)
    
    Arguments: 
            single_embedding_shape : shape of input embeddings or feature map.
                                    Must be an array.
                                    
    Output:
            A model that takes a pair of two images emneddings, concatenated
            into a single array: concatenate(img1, img2) -> single probability.
    '''
    
    #compute shape for input
    input_shape = single_embedding_shape
    #the two input embeddings will be concatenated    
    input_shape[0] = input_shape[0]*2
    
      # Neural Network
    network = Sequential(name="learned_metric")   
    network.add(Dense(30, activation='relu',
                    input_shape=input_shape, 
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer='he_uniform'))
    network.add(Dense(20, activation='relu',                   
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer='he_uniform'))  
    network.add(Dense(10, activation='relu',                   
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer='he_uniform'))
    
    #Last layer : binary softmax
    network.add(Dense(2, activation='softmax'))
    
    #Select only one output value from the softmax
    network.add(Lambda(lambda x: x[:,0]))
      
    return network


def build_quad_model(input_shape, network, metricnetwork, margin, margin2):
    '''
    Define the Keras Model for training 
    
    Arguments: 
        input_shape: shape of input images.
        network: Neural network to train outputing embeddings.
        metricnetwork: Neural network to train the learned metric.
        margin: minimal distance between Anchor-Positive and Anchor-Negative 
            for the lossfunction (alpha1).
        margin2: minimal distance between Anchor-Positive and 
            Negative-Negative2 for the lossfunction (alpha2).
        
    Ouput:
        The complete quadruplet losss model.
    
    '''
     # Define the tensors for the four input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    negative2_input = Input(input_shape, name="negative2_input")
    
    # Generate the encodings (feature vectors) for the four images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    encoded_n2 = network(negative2_input)
    
    #compute the concatenated pairs
    encoded_ap = Concatenate(axis=-1,name="Anchor-Positive")([encoded_a,encoded_p])
    encoded_an = Concatenate(axis=-1,name="Anchor-Negative")([encoded_a,encoded_n])
    encoded_nn = Concatenate(axis=-1,name="Negative-Negative2")([encoded_n,encoded_n2])
    
    #compute the distances AP, AN, NN
    ap_dist = metricnetwork(encoded_ap)
    an_dist = metricnetwork(encoded_an)
    nn_dist = metricnetwork(encoded_nn)
    
    #QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(alpha=margin,beta=margin2,name='4xLoss')([ap_dist,an_dist,nn_dist])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input,negative2_input],outputs=loss_layer)
    
    # return the model
    return network_train


def get_quad_batch(set_data, set_labels, batch_size):
    '''
    Create batch of batch_size quads, with:
        - image A: first sample of a random class A
        - image P: second sample of the same class A (different image though)
        - image N1: third image of class B with B!=A
        - image N2: fourth image of class C with C!=A and C!=B
        
    For each quad, the class is randomly drawn with replacement. 
    
    Arguments:
        set_data: set (test or train) data (numpy array)
        set_labels: set (train or test) labels (numpy array)
        batch_size: desired batch size
        
    Output:
        quads: A list of 4 np.arrays to be used for training. Each array is a 
        batch for a type of image (A, P, N1 or N2).
    '''

    n_classes=set_labels.shape[1]
    if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
        n_examples,features=set_data.shape
    else:
        n_examples,features,t,channels=set_data.shape
    
    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=True)
    
    # initialize 2 empty arrays for the input image batch
    if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
        quads=[np.zeros((batch_size, features)) for i in range(4)]
    else:
        quads=[np.zeros((batch_size, features, features, channels)) for i in range(4)]
    
    # Save actually categories for information
    actual_categories_0=np.zeros((batch_size,))
    actual_categories_1=np.zeros((batch_size,))
    actual_categories_2=np.zeros((batch_size,))
    actual_categories_3=np.zeros((batch_size,))
    actual_samples_0=np.zeros((batch_size,))
    actual_samples_1=np.zeros((batch_size,))
    actual_samples_2=np.zeros((batch_size,))
    actual_samples_3=np.zeros((batch_size,))
    for i in range(batch_size):
        
        # First image: Anchor - Class A
        category = categories[i]
        
        # subset of samples of the right category
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            subset = set_data[set_labels[:,category]==1,:]
        else:
            subset = set_data[set_labels[:,category]==1,:,:,:]
            
        nb_available_samples=subset.shape[0]
        idx_same_class = rng.choice(nb_available_samples,size=(2,),replace=False)
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            quads[0][i,:] = subset[idx_same_class[0]]
        else:
            quads[0][i,:,:,:] = subset[idx_same_class[0]]
        actual_categories_0[i]=category
        actual_samples_0[i]=idx_same_class[0]
        
       # Second image from same class
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
           quads[1][i,:] = subset[idx_same_class[1]]
        else:
           quads[1][i,:,:,:] = subset[idx_same_class[1]]
        actual_categories_1[i]=category
        actual_samples_1[i]=idx_same_class[1]
        
        # Third image from different class
        classes_left=[c for c in range(n_classes) if c!=category]
        category_different_class = rng.choice(classes_left,size=(2,),replace=False)
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            subsetB = set_data[set_labels[:,category_different_class[0]]==1,:]
        else:
            subsetB = set_data[set_labels[:,category_different_class[0]]==1,:,:,:]
        nb_available_samplesB=subsetB.shape[0]
        idx_classB = rng.randint(0, nb_available_samplesB)
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            quads[2][i,:] = subsetB[idx_classB]
        else:
            quads[2][i,:,:,:] = subsetB[idx_classB]
        actual_categories_2[i]=category_different_class[0]
        actual_samples_2[i]=idx_classB
        
        # Fourth image from another different class
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            subsetC = set_data[set_labels[:,category_different_class[1]]==1,:]
        else:
            subsetC = set_data[set_labels[:,category_different_class[1]]==1,:,:,:]
        nb_available_samplesC=subsetC.shape[0]
        idx_classC = rng.randint(0, nb_available_samplesC)
        if LooksLikeWho.SLD_models.MODEL_BASE_TAG == 'FaceNet':
            quads[3][i,:] = subsetC[idx_classC]
        else:
            quads[3][i,:,:,:] = subsetC[idx_classC]
        actual_categories_3[i]=category_different_class[1]
        actual_samples_3[i]=idx_classC
    
    return quads


def generate_quad(set_data, set_labels, batch_size):
    '''
    A generator for batches, compatible with model.fit_generator.
    
    Arguments:
        set_data: set (test or train) data (numpy array)
        set_labels: set (train or test) labels (numpy array)
        batch_size: desired batch size
        
    Output:
        quads: A list of 4 np.arrays to be used for training. Each array is a 
        batch for a type of image (A, P, N1 or N2).

    '''

    while True:
        quads = get_quad_batch(set_data, set_labels, batch_size)
        yield quads
        

def compute_learned_dist_one_vs_all(network, metricnetwork, k, train_data, \
                                    train_labels, test_data=None, test_labels=None, output_labels=1, \
                                        also_get_loss=0, verbose = 1, label="realistic", method="max", \
                                            predict=0, predict_model_name=None, image=None):
    
    '''
    This function computes the distance (similarity) between one image, either
    randomly selected from train_data or provided using the'image' argument.
    
    Arguments:
        network: encoder network
        metricnetwork: similarity function
        k: number of tests, should be 1 if predict is 1
        train_data: train data (numpy array), the reference image will be compared
            to all images contained in that array
        train_labels: train labels (numpy array)
        test_data: test data (numpy array), the reference image will be drawn
            amongst images contained in that array
        test_labels: test labels (numpy array)
        output_labels: option to print out labels when testing
        also_get_loss: option to also compute a classic binary cross entripy loss
        verbose: option to print out details about the execution of the code
        label: name of the type of testing done. Only use for console prints.
        method: "max" or "average". Decide how the predicted class will be computed,
            by either selecting the class corresponding to the image with the smallest
            distance ("min"), or by selectong the class whose top 3 matches have the
            smallest average ("average").
        predict: 0 if the function is used for in-training testing,
            1 if the function is used for predictions
        predict_model_name: if predict is 1, name of the 1st encoder. This is
            to avoid the use of too many global variables.
        image: a np array representation of an input image, when predict is 1
        
        
    Outputs:
        if predict = 1:
            predicted_cat: the predicted classs
            distance: the corresponding distance
            actual_image_index: the index of the matchig image in the train data  
            sorted_predicted_cats: an array of all the predicted_cats, sorted
                by best match to worse match
            sorted_distances: an array of all the predicted distances, sorted
                by best match to worse match
            sorted_actual_image_index: an array of all images indexes, sorted
                by best match to worse match
        if predict = 0:
                percent_correct: a custom metrics. For each one of the k tests,
                    the ground truth class is compared to its position in the 
                    sorted predicted classes of the model. For instance, if the
                    grond truth class is 3, and the model predicts 2, 3, 4, 1,
                    the percent correct will be 75%. That number is averagged
                    over all k examples. It gives an idea if a model is improving
                    or not, as it's a more granular metric that the exact_match
                    one below.
                loss: binary cross entropy loss
                exact_matches: out of the k tests, the percentage of predictions
                    that were exact.
        
    '''
    
    if predict and k!=1:
        raise Exception("Cannot predict on more than one sample.")
    
    n_correct = 0
    if verbose:
        print("Evaluating model with ({}) 1 test sample vs. all train samples\
              using the {} method...".format(str(k), method))
        
    if also_get_loss:
        bce = BinaryCrossentropy()
        loss=0
        
    rk_pct_total=0
    
    if not predict:
        print("Rounds completed:", end="\n")
        
    for i in range(k):
        gc.collect()
        if predict:
            pairs, actual_categories = make_oneshot_task_realistic(train_data,\
                train_labels, output_labels=1, predict=1, \
                predict_model_name=predict_model_name, image=image)
        else:
            pairs, targets, actual_categories = make_oneshot_task_realistic( \
                train_data, train_labels, test_data, test_labels, \
                    output_labels=1, predict=0)
        gc.collect()
        
        # Get embeddings for the test image
        test_image_embeddings=network.predict(np.expand_dims(pairs[0][0], axis=0))
        
        # Create an array to store all embeddings
        m = pairs[0].shape[0] # number of comparison to make
        embeddingsize = test_image_embeddings.shape[1]
        embeddings = np.zeros((m, embeddingsize*2))
        
        train_set_embeddings=network.predict(pairs[1])
        embeddings[:,embeddingsize:]=train_set_embeddings
        embeddings[:,:embeddingsize]=test_image_embeddings
        
        # Get distances
        distances = metricnetwork(embeddings)
        distances=np.array(distances)
        # print(type(distances))
        # print(distances.shape)
        last_correct=False
        del embeddings
        del pairs
        
        if method=="min":
            if not predict:
                if np.argmin(distances) in np.argwhere(targets == np.amax(targets)):
                    n_correct+=1
                    last_correct=True
            elif predict:
                arg_min_d=np.argmin(distances)
                predicted_cat=int(actual_categories[arg_min_d])
                distance=np.amin(distances)
                actual_image_index=arg_min_d # No need to invoke ORDERS, train not shuffled for predict
                
                # Rank all results
                sorted_actual_image_index = np.argsort(distances)
                print(type(sorted_actual_image_index))
                print(sorted_actual_image_index)
                sorted_distances = distances[sorted_actual_image_index]
                sorted_predicted_cats = actual_categories[sorted_actual_image_index].astype(int)
                
        elif method=="average":
            # Compute the average per class of the smallest 3 distances
            avg_per_class=np.zeros(len(np.unique(actual_categories)))
            unsorted_actual_image_index=np.zeros(len(np.unique(actual_categories)))
            print_i=0
            s_dist = np.argsort(distances) # <--- sort only one time the whole array
            for i in range(avg_per_class.shape[0]):
                mask=actual_categories==i
                sorted_absolute_arguments_this_class=s_dist[mask[s_dist]]
                unsorted_actual_image_index[i]=int(sorted_absolute_arguments_this_class[0])
                sorted_distances_this_class=distances[s_dist][mask[s_dist]]
                avg_per_class[i]=np.average(sorted_distances_this_class[:3])
                if print_i <= 30:
                    # print(mask)
                    # print(sorted_absolute_arguments_this_class)
                    # print(sorted_distances_this_class)
                    # print(avg_per_class[i])
                    print_i+=1
                
            sorted_predicted_cats = np.argsort(avg_per_class) # <--- categories where the average is the lowest
            sorted_actual_image_index=unsorted_actual_image_index[sorted_predicted_cats].astype(int) # <--- absolute index of the image with the lowest distance for a given class
            sorted_distances = avg_per_class[sorted_predicted_cats] # <--- 

            predicted_cat = int(np.argmin(avg_per_class))
            distance=np.min(distances[actual_categories==predicted_cat])
            if predict:
                actual_image_index = np.where(np.logical_and(actual_categories==predicted_cat, distances==distance))[0][0] # No need to invoke ORDERS, train not shuffled for predict
            if not predict:
                rk_array = avg_per_class.argsort()
                target_cat = int(actual_categories[np.argmax(targets)])
                rk_pct=100*np.where(rk_array==target_cat)[0][0]/avg_per_class.shape[0]
                rk_pct_total+=rk_pct
                print("Rank percentage =", round(100-rk_pct,2), end=" ")
                if predicted_cat == target_cat:
                    n_correct+=1
                    last_correct=True
                else: # not correct
                    pass #no further action needed
                
        else:
            raise Exception("Wrong selection technique.")
        
        if predict:
            print('SUMMARY OF PREDICTIONS')
            print("Predicted single cat:", predicted_cat, "Predicted single distance:", \
                  distance, "Predicted single index:", actual_image_index, \
                  "##############################", "Predicted categories:", \
                      sorted_predicted_cats, "Predicted distances:", sorted_distances, \
                          "Predicted indexes:",  sorted_actual_image_index, \
                      sep="\n")
            # NOTE:
            # In case of the AVERAGE technique:
                # distance is the minimum distance within the predicted class
                # sorted_distance is the sorted AVERAGE distance per class
                # Therefore, it is normal than distance !=sorted_distance[0]
            return predicted_cat, distance, actual_image_index, sorted_predicted_cats, sorted_distances, sorted_actual_image_index
            
        if also_get_loss:
            probs=1-distances
            new_loss=bce(targets, probs).numpy()
            loss+=new_loss
            
        del probs, targets, actual_categories
        
        #During testing, this allows to quickly see how accurate the model is.
        if last_correct:
            print("o")
        else:
            print("x")
    print(" ")
            
    exact_matches = (100.0 * n_correct / k)
    percent_correct = 100-rk_pct_total/k
    
    if verbose:
        if label:
            print("Got an average of {}% realistic exact matches one-shot learning accuracy on the {} set over {} repetitions.\n".format(exact_matches,label,k))
        else:
            print("Got an average of {}% realistic exact matches one-shot learning accuracy \n".format(exact_matches))            
    
    if method=="average":
        print("The average scoring is {}% (0% is best, 100% is worst).".format(round(percent_correct,0)))
        
    if also_get_loss:
        loss=loss/k
        return percent_correct, loss, exact_matches
    else:
        return percent_correct

# END OF CODE