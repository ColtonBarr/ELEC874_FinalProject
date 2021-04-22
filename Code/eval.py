'''
This file encapsulates the training and evaluation script for 
all the models.
'''

import numpy as np
import pandas as pd
import argparse
import HelperFxns as fxns
import os, sys

from Classifiers import Classifier1


def evaluate(general_params):

    #Initialize a list to append results to
    output = []
    output.append(" ---------------------------------- ")
    output.append("\n ------- Evaluation Results ------- ")
    output.append("\n ---------------------------------- ")
    output.append("")

    #Read in the data csv
    data_df = pd.read_csv(general_params['data_csv']) 

    #Iterate through all folds
    for fold_num in range(general_params['num_folds']):

        #Determine the directory for the current fold
        fold_dir = general_params['save_path']+"\\Fold_"+str(fold_num)

        #Iterate through each model
        for model in general_params['models']:

            model.evaluate(fold_dir, fold_num, data_df)


def main():

    #Instantiate all classifier objects
    models = [Classifier1()]

    #Parse flags
    data_csv = FLAGS.data_csv
    output_folder = FLAGS.model_path
    num_folds = int(FLAGS.num_folds)

    #Get all model filenames to use
    model_flag = int(FLAGS.model)
    if model_flag != -1:
        models = models[model_flag + 1]

    general_params = {"save_path"     : output_folder,
                      "num_folds"     : num_folds,
                      "data_csv"      : data_csv,
                      "models"        : models}

    evaluate(general_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_csv',
        type=str,
        default="C:\\repos\\Courses\\ELEC874_FinalProject\\Code\\ELEC874_SampleDataset.csv",
        help='Path to the csv file containing locations for all data used in training'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="C:\\repos\\Courses\\ELEC874_FinalProject\\NetworkOutputs\\Test_0",
        help='Path to the directory that contains the trained networks for each fold.'
    )   
    parser.add_argument(
        '--num_folds',
        type=str,
        default='1',
        help='Number of folds used in the K fold cross validation scheme.'
    )  
    parser.add_argument(
        '--model',
        type=str,
        default='-1',
        help='The number of the specific model that you wish to use.'
    )  

    #Get input flag arguments
    FLAGS, unparsed = parser.parse_known_args()

    main()