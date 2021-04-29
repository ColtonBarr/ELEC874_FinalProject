import os
import sys
import numpy as np
import pandas as pd
import HelperFxns as fxns
from pathlib import Path
import cv2
import collections
import random
from numpy import newaxis
from tqdm import tqdm

import tensorflow
import tensorflow.keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Input, Dropout, Add, BatchNormalization
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
import sklearn.metrics
from scipy.ndimage.interpolation import rotate, zoom

'''--- Classifier Class -- 
This is the class that will generate each possible architecture used in this experiment.

It will take advantage of a model building method that can customize the specific architecture,
as well as training and evaluation methods that work with any generic classifier.
'''

class Classifier():

    def __init__(self, prefix="CNN"):

        #Establish default hyperparameters that child models can tweak
        self.params = {"num_epochs"     : 10,
                        "batch_size"    : 8,
                        "learning_rate" : 0.0001,
                        "optimizer"     : tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
                        "loss_fxn"      : 'categorical_crossentropy',
                        "metrics"       : ['accuracy']}

        #Instantiate class variables
        self.prefix = prefix
        self.aug = None
        self.deep = None
        self.bnorm = None
        self.drop = None
        self.skip = None


    #The core block of the model. Note that the default is only using
    #one of these blocks, but if the "deep" keyword is set we use two.
    def add_block(self, input_layer):

        #First convolutional block
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    
        #Add optional parts if indicated
        if self.bnorm:
            x = BatchNormalization()(x)
        if self.drop:
            x = Dropout(0.25)(x)

        #Second convolutional block
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

        #Add optional parts as neccesary
        if self.bnorm:
            x = BatchNormalization()(x)
        if self.drop:
            x = Dropout(0.25)(x)

        #If residual connections are being used, reduce the channels to 1 before
        #adding the input layer
        if self.skip:
            x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
            x = Add()([x, input_layer])

        #Max pooling layer
        pooled = MaxPool2D(3, 3)(x)

        return pooled    
        

    #Returns the classifier model
    def create_model(self):

        #Define input layer
        input_layer = Input(shape=(128, 128, 1))

        #Add the main classifier structure
        x = self.add_block(input_layer)

        #If the deep hyperparameter it set, add two more blocks.
        if self.deep:
            x = self.add_block(x)
            x = self.add_block(x)

        #Flatten the output and feed it through a single
        #dense layer before softmax activation.
        fl = Flatten()(x)
        d1 = Dense(32, activation='relu')(fl)
        d2 = Dense(3, activation='softmax')(d1)

        model = tensorflow.keras.Model(
            inputs=input_layer,
            outputs=d2,
            name=self.name
        )

        return model

    #Load the model from the saved file
    def load_model(self, save_folder):
        return tensorflow.keras.models.load_model(save_folder)

    #Performs a random flip across the y axis
    def random_flip(self, image):
        if random.choice([0,1]):
            return cv2.flip(image, 1)
        return image

    #Randomly dims the image
    def random_brightness(self, image):
        value = random.choice([1, 2, 5, 10, 20])
        return np.where((0 + image) < value,0,image-value)

    #Returns an image that has been randomly augmented
    def augment_image(self, image):
        rot_img = self.random_brightness(self.random_flip(image))
        return rot_img

    #Load in the data for training, validation and testing
    def load_data(self, fold, section, data_df):
        entries = data_df.loc[(data_df["Fold"] == fold) & (data_df["Set"] == section)]
        images = []
        imageLabels = []

        #If augmentation is used, this is the # of additional images
        num_aug = 3

        cwd = Path(os.getcwd()).parent.absolute()

        for i in tqdm(entries.index):

            #Read in the ultrasound image
            file_dir = entries["Folder"][i]
            filename_us = entries['US_Image'][i]
            full_path_us = os.path.join(cwd, file_dir, "Ultrasound", filename_us)
            image = cv2.imread(full_path_us, 0)

            #Process the original US image and add to the dataset
            processed_us = self.process_ultrasound(image)
            images.append(np.array(processed_us))
            imageLabels.append(entries['US_Pose'][i])

            #If "aug" flag is set, add 3 more augmented images
            if self.aug:

                for j in range(num_aug):

                    aug_image = self.augment_image(image)
                    processed_us = self.process_ultrasound(aug_image)
                    images.append(np.array(processed_us))
                    imageLabels.append(entries['US_Pose'][i])

        return np.array(images), np.array(imageLabels)

    #Perform resizing and scaling of the input US images
    def process_ultrasound(self, image):
        resized = cv2.resize(image, (128, 128)).astype(np.float16)
        scaled = resized / resized.max()
        return scaled[...,np.newaxis]

    #Save the model to a file
    def save_model(self, model, save_folder):
        full_model_path = os.path.join(save_folder, self.name)
        model.save(full_model_path)
        return

    #Loads the specific data used for the current fold
    def load_fold_data(self, fold_num, data_df, aug=False):

        self.aug = aug

        #Read in all data for training, validation and testing
        self.train_imgs, self.train_labels = self.load_data(fold_num, "Train", data_df)
        self.val_imgs, self.val_labels = self.load_data(fold_num, "Validation", data_df)
        self.test_imgs, self.test_labels = self.load_data(fold_num, "Test", data_df)

        #onehot encode the labels
        encoder = OneHotEncoder(handle_unknown='ignore')
        self.train_labels = encoder.fit_transform(self.train_labels.reshape(-1,1)).toarray()
        self.val_labels = encoder.fit_transform(self.val_labels.reshape(-1,1)).toarray()
        self.test_labels = encoder.fit_transform(self.test_labels.reshape(-1,1)).toarray()

    #Takes the current class variables for the network properties and generates 
    #a string that names the architecture.
    def generate_model_name(self):
        flag_strs = ['aug','deep','bnorm','drop','skip']
        flag_bools = [self.aug, self.deep, self.bnorm, self.drop, self.skip]

        suffix = '_'.join(tuple([flag_strs[i] for i in range(5) if flag_bools[i]]))

        return self.prefix + "_" + suffix
    
    #The main training method. Generates a model based on model_params followed by
    #training, validation and testing the model.
    def train(self, fold_params, model_params):

        #Set flags as the model wide parameters
        self.deep = model_params['deep']
        self.bnorm = model_params['bnorm']
        self.drop = model_params['drop']
        self.skip = model_params['skip']

        #Create a model that matches these specifications
        self.name = self.generate_model_name()

        #Start by creating a directory to hold network outputs
        output_dir = fold_params['fold_dir'] + "\\" + self.name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)        

        #Create the model that was specified using "model_params"
        model = self.create_model()
        print(model.summary())

        #Compile and fit the model.
        model.compile(optimizer = self.params['optimizer'], loss = self.params['loss_fxn'], metrics = self.params['metrics'])        
        history = model.fit(x=self.train_imgs,
                            y=self.train_labels,
                            validation_data=(self.val_imgs,self.val_labels),
                            batch_size = self.params['batch_size'],
                            epochs = self.params['num_epochs'],
                            callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])

        #Test the model
        results = model.evaluate(x = self.test_imgs,
                                    y = self.test_labels,
                                    batch_size = self.params['batch_size'])

        self.save_model(model, output_dir)

        #Get network predictions on the test set
        predictions = model.predict(x = self.test_imgs)
        predictions = np.argmax(predictions, axis=-1)
        test_labels_argmax = np.argmax(self.test_labels, axis=-1)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_labels_argmax, predictions)

        #Save output info to quantify performance.
        fxns.save_training_info(output_dir, fold_params['fold_num'], self.params, history.history, results, self.name, confusion_matrix)
        fxns.save_training_plot(output_dir, history.history, "loss", self.name)

        for metric in self.params["metrics"]:
            fxns.save_training_plot(output_dir, history.history, metric, self.name)
 
    #Strictly test the network and return the raw results.
    def evaluate(self, model):

        #Get network predictions on the test set
        predictions = model.predict(x = self.test_imgs)
        evaluation = model.evaluate(x = self.test_imgs,
                                    y = self.test_labels,
                                    batch_size = self.params['batch_size'])
        accuracy = evaluation[1]

        #Process predictions
        predictions = np.argmax(predictions, axis=-1)
        test_labels = np.argmax(self.test_labels, axis=-1)

        #Generate the confusion matrix and overall report
        confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)
        overall_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=["Cross-section", "Long-axis", "Undefined"] , digits=4)

        return accuracy, confusion_matrix, overall_report
    
       