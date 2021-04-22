'''
The purpose of this file is to encapsulate classifier #1.

This is the base classifier to which all subsequent classifiers
will be compared.
'''
import os
import sys
import numpy as np
import pandas as pd
import HelperFxns as fxns
from pathlib import Path
import cv2
import collections

import tensorflow
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Input, GlobalAveragePooling2D, Lambda
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import sklearn
import sklearn.metrics

###############################################################################
# --- Base Classifier Class -- 
# This is the object that all other classifiers will inherit from.
# They can reimplement or extend any methods and parameters that are
# unique in that given implementation.

class BaseClassifier():

    def __init__(self, name):

        self.name = name

        #Establish default hyperparameters that child models can tweak
        self.params = {"num_epochs" : 10,
                        "batch_size"    : 8,
                        "learning_rate" : 0.0001,
                        "optimizer"     : tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
                        "loss_fxn"      : 'categorical_crossentropy',
                        "metrics"       : ['accuracy']}


    #Returns the classifier model
    def create_model(self):

        #Input layer; takes in US image
        input_layer = Input(shape=(128, 128, 1))

        cov2d_1 = Conv2D(8, (3, 3), activation='relu')(input_layer)  
        cov2d_2 = Conv2D(8, (3, 3), activation='relu')(cov2d_1)  
        pl = MaxPool2D((3, 3))(cov2d_2)   

        fl = Flatten()(pl)
        d1 = Dense(64, activation='relu')(fl)
        d2 = Dense(3, activation='softmax')(d1)

        #Build the model from the input and output layers
        model = tensorflow.keras.Model(
            inputs=input_layer,
            outputs=d2,
        )
        return model


    #Load the model from the saved file
    def load_model(self, save_folder):
        full_model_path = os.path.join(save_folder, self.name, self.name)
        return tensorflow.keras.models.load_model(full_model_path)


    #Load in the data for training, validation and testing
    def load_data(self, fold, section, data_df):
        entries = data_df.loc[(data_df["Fold"] == fold) & (data_df["Set"] == section)]
        images = []
        imageLabels = []

        cwd = Path(os.getcwd()).parent.absolute()

        for i in entries.index:

            #Read in the ultrasound image
            file_dir = entries["Folder"][i]
            filename_us = entries['US_Image'][i]
            full_path_us = os.path.join(cwd, file_dir, "Ultrasound", filename_us)
            image = cv2.imread(full_path_us, 0)
            processed_us = self.process_ultrasound(image)
            images.append(np.array(processed_us))

            #Read in the segmentation image
            imageLabels.append(entries['US_Pose'][i])

        return np.array(images), np.array(imageLabels)

    def process_ultrasound(self, image):
        resized = cv2.resize(image, (128, 128)).astype(np.float16)
        scaled = resized / resized.max()
        return scaled[...,np.newaxis]

    def save_model(self, model, save_folder):
        full_model_path = os.path.join(save_folder, self.name)
        model.save(full_model_path)
        return
    
    
    def train(self, fold_dir, fold_num, data_df):

        #Start by creating a directory to hold network outputs
        output_dir = fold_dir + "\\" + self.name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)        

        #Read in all data for training, validation and testing
        train_imgs,train_labels = self.load_data(fold_num, "Train", data_df)
        val_imgs,val_labels = self.load_data(fold_num, "Validation", data_df)
        test_imgs,test_labels = self.load_data(fold_num, "Test", data_df)

        print("-------------------------------------------")
        print("Train: {}".format(collections.Counter(train_labels)))
        print("Val: {}".format(collections.Counter(val_labels)))
        print("Test: {}".format(collections.Counter(test_labels)))

        #onehot encode the labels
        encoder = OneHotEncoder(handle_unknown='ignore')
        train_labels = encoder.fit_transform(train_labels.reshape(-1,1)).toarray()
        val_labels = encoder.fit_transform(val_labels.reshape(-1,1)).toarray()
        test_labels = encoder.fit_transform(test_labels.reshape(-1,1)).toarray()

        model = self.create_model()
        print(model.summary())

        model.compile(optimizer = self.params['optimizer'], loss = self.params['loss_fxn'], metrics = self.params['metrics'])        
        history = model.fit(x=train_imgs,
                            y=train_labels,
                            validation_data=(val_imgs,val_labels),
                            batch_size = self.params['batch_size'],
                            epochs = self.params['num_epochs'],
                            callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])

        results = model.evaluate(x = test_imgs,
                                    y = test_labels,
                                    batch_size = self.params['batch_size'])

        self.save_model(model, output_dir)

        #Get network predictions on the test set
        predictions = model.predict(x = test_imgs)
        predictions = np.argmax(predictions, axis=-1)
        test_labels = np.argmax(test_labels, axis=-1)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)

        fxns.save_training_info(output_dir, fold_num, self.params, history.history, results, self.name, confusion_matrix)

        fxns.save_training_plot(output_dir, history.history, "loss", self.name)

        for metric in self.params["metrics"]:
            fxns.save_training_plot(output_dir, history.history, metric, self.name)

    def evaluate(self, save_dir, fold_num, data_df):

        #Read in data for testing
        test_imgs,test_labels = self.load_data(fold_num, "Test", data_df)

        print("-------------------------------------------")
        print("Test: {}".format(collections.Counter(test_labels)))

        #onehot encode the labels
        encoder = OneHotEncoder(handle_unknown='ignore')
        test_labels = encoder.fit_transform(test_labels.reshape(-1,1)).toarray()

        #Load the model
        model = self.load_model(save_dir)
        print(model.summary())

        #Get network predictions on the test set
        predictions = model.predict(x = test_imgs)

        #Process predictions
        predictions = np.argmax(predictions, axis=-1)
        test_labels = np.argmax(test_labels, axis=-1)
        print("test labels: " + str(test_labels))
        print("predictions: " + str(predictions))
        confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)
        overall_report = sklearn.metrics.classification_report(test_labels, predictions, target_names=["Cross-section", "Long-axis", "Undefined"] , digits=4)

        print("---------------------------------------------------------------------")
        print(" *** Classifier: " + self.name + " *** ")
        print("Fold #" + str(fold_num) + " Results\n")
        print(overall_report)
        print("\n Confusion Matrix: \n")
        print(confusion_matrix)
        print("\n---------------------------------------------------------------------")        
       

###############################################################################
# -- Classifier 1 -- 
# The default implementation of the classifier. 
# Makes zero modifications to the base classifier beyond assigining a name

class Classifier1(BaseClassifier):

    def __init__(self):
        super().__init__("Classifier1")



# class BaseObj():

#     def __init__(self, name):

#         self.name = name
#         self.hyperparams = None

# class ChildObj(BaseObj):

#     def __init__(self):
#         super().__init__("Classifier1")

#         self.hyperparams = {"new" : "params"}