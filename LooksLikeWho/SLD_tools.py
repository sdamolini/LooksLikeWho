# -*- coding: utf-8 -*-
'''
This module contains helper functions.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
'''

# IMPORTS

import pandas as pd
import numpy as np
import numpy.random as rng
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import PIL
import cv2
import mtcnn
import gc
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os
import shutil
import random
from glob import glob
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from itertools import cycle
from tensorflow.keras.losses import BinaryCrossentropy

import LooksLikeWho.SLD_models


# FUNCTIONS

def populate_classes(data_path, model_name, dataset_tag):
    '''
    Obtain the list of classes and saves it as .npy for later use on AWS.
    
    Arguments:
        data_path: the folder where the 'train' and 'test' folders are.
        model_name: string corresponding to the model.
        dataset_tag: name of the current dataset. 
                
    Outputs:
        Two .npy files written to the disk.
    '''
    
    global classes
    global classes_mapping
    
    classes_path=os.path.join('../models', 'bottlenecks',  model_name + dataset_tag + '_CLASSES.npy')
    classes_path_mapping=os.path.join('../models', 'bottlenecks', model_name + dataset_tag + '_CLASSES_MAPPING.npy')
    
    if os.path.exists(classes_path) and os.path.exists(classes_path_mapping):
        classes = np.load(classes_path, allow_pickle=True)
        classes_mapping = np.load(classes_path_mapping, allow_pickle=True)
    else:
        # Identify labels based on the available folders in the train and test set
        train_test_labels=get_labels(data_path)
        classes = train_test_labels[0]  
        classes_mapping = dict(enumerate(classes))
        
        # Save classes for AWS use
        np.save(classes_path, classes)
        np.save(classes_path_mapping, classes_mapping)
        
        
def make_oneshot_task_realistic(train_data, train_labels, test_data=None, \
                                test_labels=None, output_labels=0, predict=1, \
                                    predict_model_name=None, image=None):
    '''
    Create batch of pairs, one sample from the test set versus ALL samples in ALL classes of train set. 
    
    Arguments:
        train_data: train data (numpy array), the reference image will be compared
            to all images contained in that array
        train_labels: train labels (numpy array)
        test_data: test data (numpy array), the reference image will be drawn
            amongst images contained in that array, unless 'image' is provided. 
        test_labels: test labels (numpy array)
        output_labels: option to print out labels when testing
        predict: 0 if the function is used for in-training testing,
                 1 if the function is used for predictions
        predict_model_name: if predict is 1, name of the 1st encoder. This is
            to avoid the use of too many global variables.
        image: a np array representation of an input image, when predict is 1
                
    Outputs:
        pairs, actual_categories_1 if predict = 1
        pairs, targets if predict = 0 and output_labels = 1
        pairs, targets, actual_categories_1 if predict = 0 and output_labels = 0
        with:
            pairs: list of two tensors, one for the tested image (repeated as
                needed), one for the train set.
            targets: when the class of the tested image is known, target is a
                vector containing 1 where the train set is of the same class, 
                0 otherwise.
            actual_categories_1: returns classes for the train set.
    '''

    # Obtain the model name differently if running for prediction or not
    if predict:
        model_name = predict_model_name
    else:
        model_name = LooksLikeWho.SLD_models.MODEL_BASE_TAG
    
    if len(train_labels.shape)==1:
        # AWS one dimension format
        n_classes=np.unique(train_labels).shape[0]
    else:
        # dense matrix format
        n_classes=train_labels.shape[1]
        
    if model_name == 'FaceNet':
        # print(train_data.shape)
        n_examples,features=train_data.shape
    else:
        n_examples,features,t,channels=train_data.shape
      
    # Select the category whose sample is going to be drawn from
    category = rng.randint(0,n_classes)
    
    # initialize 2 empty arrays for the input image batch
    if model_name == 'FaceNet':
        pairs=[np.zeros((n_examples, features)) for i in range(2)]
    else:
        pairs=[np.zeros((n_examples, features, t, channels)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((n_examples,))
    
    # Save actually categories for information
    actual_categories_0=np.zeros((n_examples,))
    actual_categories_1=np.zeros((n_examples,))
    
    # Targets are one for same class.
    if len(train_labels.shape)==1:
        # AWS one dimension format
        targets[train_labels==category]=1
    else:
        # dense matrix format
        targets[train_labels[:,category]==1]=1
    
    # Select a random test image from the selected category
    if not predict:
        if model_name == 'FaceNet':
            subset0_test = test_data[test_labels[:,category]==1,:]
        else:
            subset0_test = test_data[test_labels[:,category]==1,:,:,:]
        nb_available_samples0_test=subset0_test.shape[0]
        idx_1_test = rng.randint(0, nb_available_samples0_test)
        sample_image = subset0_test[idx_1_test]
    elif predict:
        sample_image=image
        
    if model_name == 'FaceNet':
        pairs[0][:,:] = sample_image
        actual_categories_0[:] = category
        # actual_id_0[:] = idx_1_test
        
        pairs[1][:,:] = train_data
        if len(train_labels.shape)==1:
            # AWS one dimension format
            actual_categories_1[:] = train_labels
        else:
            # dense matrix format
            actual_categories_1[:] = np.argmax(train_labels, axis=1)
        # actual_id_1[:] = 
    else:
        pairs[0][:,:,:,:] = sample_image
        actual_categories_0[:] = category
        # actual_id_0[:] = idx_1_test
        
        pairs[1][:,:,:,:] = train_data
        if len(train_labels.shape)==1:
            # AWS one dimension format
            actual_categories_1[:] = train_labels
        else:
            # dense matrix format
            actual_categories_1[:] = np.argmax(train_labels, axis=1)
        # actual_id_1[:] = 

    if predict:
        return pairs, actual_categories_1
    
    if output_labels==0:
        return pairs, targets
    elif output_labels==1:
        return pairs, targets, actual_categories_1


def get_labels(data_path, mapfile="identity_meta.csv"):
    '''
    Obtains the match between the folder in the train/ test folders and their labels.
    
    Arguments:
        data_path: the folder where the 'train' and 'test' folders are
        mapfile: the name of the file with the mapping, located in data_path

    Outputs:
        res = [npar11, nparr2]: a list of two np.arrays with the labels for the train and test sets.
    '''
    res=[]
    mapfile_path=os.path.join(data_path, mapfile)
    
    for folder in ['train', 'test']:
        path=os.path.join(data_path, folder)
        ldir=np.array(os.listdir(path))
        df_mapfile=pd.read_csv(mapfile_path, sep=', ', engine='python', encoding='utf8') # <--- This is because some names have a "," in them
        res.append(np.array(df_mapfile.loc[df_mapfile['Class_ID'].map(lambda x: x in ldir),'Name'].tolist()))
        
    return res
  
    
def get_what_from_full_set_with_face_crop(what, data_path_source, data_path_dest, max_samples_per_class=1, max_classes=None, target_size = (224,224), mini_res=False):
    '''
    Browse full train AND test folders, copies samples from each class and move them to the dest train folder.
    
    Arguments:
        what: "train" or "test".
        data_path_source: the folder where the 'train' and 'test' folders are (FULL SET)
        data_path_destination: the folder where the WHAT folders are (DESTINATION)
        ratio: fraction of sample extracted from full set and moved to the train folder

    Outputs:
        A train folder in the data_path_dest.
    '''
    train_folder_source = os.path.join(data_path_source, 'train')
    test_folder_source = os.path.join(data_path_source, 'test')
    folder_dest = os.path.join(data_path_dest, what)
    count_classes=0
    other = ["train", "test"]
    other.remove(what)
    other=other[0]
    other_folder_dest = os.path.join(data_path_dest, other)
    total_count=0
    incomplete_classes=0
    gc.collect()
    
    # prepare the face detector
    detector = MTCNN()
    
    for source in [train_folder_source, test_folder_source]:
        if count_classes==max_classes:
                break
        for subf in os.listdir(source):
            if count_classes==max_classes:
                break
            count_classes+=1
            imdir = os.listdir(os.path.join(source, subf))
            random.shuffle(imdir)
            nb_samples=max_samples_per_class
            newimdir=imdir
            os.makedirs(os.path.join(folder_dest, subf) , exist_ok=True)
            # Count files already in destination
            count = len(os.listdir(os.path.join(folder_dest, subf)))
            print("Need to transfer {} samples.".format(max(0,nb_samples-count)))
            print(os.path.join(folder_dest, subf))
            if count >= nb_samples: 
                total_count+=count
                continue
            attempts=0
            native_res_factor=1
            cropped_res_factor=1
            confidence_factor=1
            while True:
                for im in newimdir:
                    if not os.path.exists(os.path.join(folder_dest, subf, im)) and not os.path.exists(os.path.join(other_folder_dest, subf, im)):
                        # impath=os.path.join(folder_dest, subf, im)
                        # shutil.copy(os.path.join(source, subf, im), impath)
                        impath=os.path.join(source, subf, im)
                        impath_target=os.path.join(folder_dest, subf, im)
                        try:
                            img = cv2.imread(impath)
                            full_height, full_width, channels = img.shape
                            if mini_res and min(full_height, full_width)<mini_res*2.2*native_res_factor:
                                # print("NATIVE RESOLUTION TOO LOW: skipping image.")
                                continue
                            detections = detector.detect_faces(img)
                            print("Detections:", detections)
                            if detections == []:
                                print("NO FACE DETECTED: skipping image {}/{}.".format(subf, im))
                                continue
                            
                            if detections[0]['confidence'] < 0.99*confidence_factor:
                                print("CONFIDENCE TOO LOW: skipping image.")
                                continue
                            x1, y1, width, height = detections[0]['box']
                            current_res=min(width, height)
                            print("Image resolution:", current_res)
                            if mini_res and current_res<mini_res*cropped_res_factor:
                                print("CROPPED RESOLUTION TOO LOW: skipping image.")
                                continue
                            # Make image square
                            w2=width//2
                            xc = x1+w2 # X centroid
                            h2=height//2
                            yc = y1+h2 # Y centroid
            
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
                                print("FACE PARTIALLY SHOWN ON ORIGINAL IMAGE (TOO ZOOMED IN): skipping image.")
                                continue
                            
                            face = img[Y0:Y1, X0:X1,::-1] #<--- Invert 1st and last channel
            
                            # resize pixels to the model size
                            face = PIL.Image.fromarray(face, mode='RGB')
                            face=face.resize(target_size, Image.BICUBIC)
                            # Remove original image
                            # os.remove(impath)
                            
                            #save croppped image
                            face.save(impath_target, "JPEG", icc_profile=face.info.get('icc_profile'))
                            del face
                            count+=1
                            total_count+=1

                            if count==nb_samples: break
                        except:
                            # delete image and try with another image
                            # os.remove(impath)
                            continue
                        
                if count==nb_samples:
                        print("******************************************")
                        print("Total Count: {} || Completion: {:.2%}.".format(total_count, total_count/(max_classes*max_samples_per_class)))
                        print("******************************************")
                        break
                    
                attempts+=1
                if attempts==1:
                    native_res_factor=1.05/2.2
                    confidence_factor=0.97/0.99
                    print("******************************************")
                    print("******************************************")
                    print("**WARNING : CHECKING MORE IMAGES *********")
                    print("******************************************")
                    print("******************************************")
                if attempts==2:
                    native_res_factor=0.9/2.2
                    cropped_res_factor=0.75
                    confidence_factor=0.95/0.99
                    print("*********************************************")
                    print("********************************************")
                    print("**WARNING : LOWER RESOLUTION MODE ENABLED **")
                    print("********************************************")
                    print("********************************************")
                if attempts==3:
                    native_res_factor=0.60/2.2
                    cropped_res_factor=0.50
                    confidence_factor=0.90/0.99
                    print("***********************************************")
                    print("***********************************************")
                    print("**WARNING : ULTRALOW RESOLUTION MODE ENABLED **")
                    print("***********************************************")
                    print("***********************************************")
                if attempts==4:
                    native_res_factor=0
                    cropped_res_factor=0.30
                    confidence_factor=0.80/0.99
                    print("***********************************************")
                    print("***********************************************")
                    print("**WARNING : LAST CHANCE ROUND !!!!!!!!!!!!!! **")
                    print("***********************************************")
                    print("***********************************************")
                if attempts==5:
                    print("Not enough good quality images.")
                    incomplete_classes+=1
                    break

    print("Incomplete clases:", str(incomplete_classes))   
    print('Done.')

    
def plot_history_quad(history, save_path, custom_file_name=""):
    '''
    Generate three plots for the siamese model:
        1. Loss
        2. Accuracy
        3. Learning Rate
        
    Arguments:
        history: a training history file
        save_path: disk path for saving resulting png file
        custom_file_name: name of the output png file (optional)
        
    Output:
        A figure representing the 'history' data, saved on disk.
    '''
    
    ## create figure
    fig, axes = plt.subplots(1,3,figsize=(16,6))
    history[['loss', 'one_shot_loss_train', 'one_shot_loss_realistic']].plot(ax=axes[0], color=['red', 'black', 'green'])
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Losses")
    axes[0].set_xticks(history.index)
    
    history[['one_shot_accuracy_train', 'one_shot_accuracy_realistic', 'one_shot_exact_matches_train', 'one_shot_exact_matches_realistic']].plot(ax=axes[1], color=['red', 'green', 'tomato', 'chartreuse'])
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("OneShotAccuracy")
    axes[1].set_xticks(history.index)
    
    history["lr"].plot(ax=axes[2], color=['green'])
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Learning Rate")
    axes[2].set_xticks(history.index)
    plt.tight_layout()
    plt.savefig(save_path+"/history"+custom_file_name+".png", dpi=150)
    plt.close('all')
    
    
def display_image(img):
    '''
    Simple function that allows multiple images to be outputted locally.
    
    Arguments:
        img: Input image as array.

    Output:
        The image is plotted in the console. 
    '''
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    
# END OF CODE