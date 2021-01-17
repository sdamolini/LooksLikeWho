# -*- coding: utf-8 -*-
"""
Flask deployed website, main code.

Author: Stephane Damolini
Site: LooksLikeWho.damolini.com 
"""

# IMPORTS

import shutil
import gc, os, sys

from flask import Flask, render_template
from flask import redirect, flash, url_for, request
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename

from app import app
import LooksLikeWho

sys.path.append('..')
sys.path.append('../LooksLikeWho')

retval = os.getcwd()
print("[>>views<<] Current working directory %s" % retval, \
      "__NAME__ = ", __name__)

############################### USER INPUTS ##################################

# PARAMETERS AND SETTING FOR THE DEPLOYED MODEL

# Select dataset (uncomment one)
# dataset_tag = "_litecropped" # 2 x 10 x 4
# dataset_tag = "_medium_cropped" # 2 x 100 x 4
# dataset_tag = "_mediumcroppedx10"# 2 x 100 x 10
# dataset_tag = "_ALL-HQ-UNZOOMED" 2 x 1931 x 4
dataset_tag = "_ALL-HQ-UNZOOMED-10X" # 10931 x 10 (train) + 1931 x 4 (test)

# Set parameters
method="min"            # "average" or "min". Method of chsosing winning class
model_name="FaceNet"    # Name of the pretrained encoder
from_notebook=False     # True if this code is executed alone
online=True             # True if dataset located within app folder

########################## END OF USER INPUTS ################################


# LOADING MODEL AND DATA
base_model, \
network, \
metricnetwork, \
data_path, \
train_data, \
train_labels_AWS,\
train_filenames, \
classes, \
target_size = LooksLikeWho.SLD_models.load_models_and_data(model_name, \
                                    dataset_tag, from_notebook, online=online)
gc.collect()


# CREATE FLASK APP
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=16,
    DROPZONE_MAX_FILES=1,
    DROPZONE_DEFAULT_MESSAGE="<img src=\"/static/cloud.png\" width=\"30\"> " +
    "Drop a file, or Browse.",
    DROPZONE_MAX_FILE_EXCEED="Your can't upload any more files.",
    DROPZONE_REDIRECT_VIEW='predict')
dropzone = Dropzone(app)


# OTHER PARAMETERS
## Valid extensions checked before upload
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## Folders
UPLOAD_FOLDER = 'app/static/uploads/'
STYLE_FOLDER = 'app/static/styles'
PREDICTION_FOLDER = 'app/static/prediction'
SAMPLES_FOLDER = 'app/static/samples'
DATASET_TAG = dataset_tag
ALL_DATASETS_FOLDER = r'datasets/'
DATASET_FOLDER = ALL_DATASETS_FOLDER+DATASET_TAG

## Update paths if needed
def from_notebook():
    global UPLOAD_FOLDER
    global STYLE_FOLDER
    global PREDICTION_FOLDER
    global DATASET_FOLDER
    global SAMPLES_FOLDER
    UPLOAD_FOLDER = '../'+UPLOAD_FOLDER
    STYLE_FOLDER = '../'+STYLE_FOLDER
    DATASET_FOLDER = '../'+DATASET_FOLDER
    PREDICTION_FOLDER = '../'+PREDICTION_FOLDER
    SAMPLES_FOLDER = '../'+SAMPLES_FOLDER
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['STYLE_FOLDER'] = STYLE_FOLDER
    app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
    app.config['DATASET_FOLDER'] = DATASET_FOLDER
    app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DATASET_TAG'] = DATASET_TAG

## Function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() \
        in ALLOWED_EXTENSIONS


# DEFINE PAGES
@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            ## clear cache if needed
            files = os.listdir(app.config['UPLOAD_FOLDER'])
            for f in files:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

            ## get uploaded file
            filename = secure_filename(file.filename)

            ## save uploaded file
            print('Saving uploaded file...')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('Copying uploaded file...')
            shutil.copyfile(os.path.join(app.config['UPLOAD_FOLDER'], \
                filename), os.path.join(app.config['SAMPLES_FOLDER'], \
                filename))
            return render_template('home.html')
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    else:
        return render_template('home.html')

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
        retval = os.getcwd()
        print("Current working directory %s" % retval)  
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) >0:
            filename = os.listdir(app.config['UPLOAD_FOLDER'])[0]
        else:
            filename = None

        if filename:
             ## remove existing predictions
            files = os.listdir(app.config['PREDICTION_FOLDER'])
            for f in files:
                os.remove(os.path.join(app.config['PREDICTION_FOLDER'], f))

            ## make predictions
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filename_noext=os.path.splitext(filename)[0]
            try:
                pred_class, distance, actual_image_index, \
                    actual_image_path, matches_list = \
                    LooksLikeWho.SLD_models.make_prediction_quad(model_name, \
                        base_model, network, metricnetwork, image_path, \
                        train_data, train_labels_AWS, train_filenames, \
                        classes, data_path, target_size, dataset_tag, \
                        filename=filename_noext, method="min", \
                        extra_matches=5, online=True)
            except Exception as e:
                print(e)
                flash('Please wait about 20 seconds for the prediction! ' \
                      '92,000 photos are being compared to yours!')
                flash('Sorry, the algorithm could not detect a face with' \
                      ' accuracy, please try with another picture (front ' \
                          'facing is best!).')
                return render_template('home.html')
                
            celeb_name = matches_list[0].name
            matches_list=matches_list[1:] # Remove 1st match we already have
            
            gc.collect()

            match_result_text="Your match is {} (pictured on the right)! " \
                "See below for additional matches out of the 9131 " \
                    "celebrities in the database!".format(celeb_name)
            plot = 'prediction_{}.png'.format(filename_noext)
            print('BEFORE RENDER')
            return render_template('home.html', filename=filename, \
                    celeb_name=celeb_name, match_result_text= \
                    match_result_text, plot=plot, samples=matches_list)
        else:
            return render_template('home.html')
    
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), \
                    code=301)

@app.route('/plot/<filename>')
def display_plot(filename):
    print('plot filename: ' + filename)
    return redirect(url_for('static', filename='prediction/' + filename), \
                    code=301)

@app.route('/display_train_sample/<sample_path>')
def display_train_sample(sample_path):
    dataset_tag
    folder, filename = sample_path.split("____")
    print('FILENAME SPLITTED')
    print(folder, filename)
    return redirect(url_for('static', filename= 'datasets/' + app.config[\
            'DATASET_TAG'] + '/train/' + folder + '/' + filename), code=301)

@app.route('/about')
def about():
    return render_template('about.html')    

@app.route('/model')
def model():
    return redirect("http://LooksLikeWho.damolini.com", code=302)

# END OF CODE