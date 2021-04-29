'''
This file encapsulates the training script for 
all the models.
'''

import numpy as np
import pandas as pd
import argparse
import HelperFxns as fxns
import os, sys
import itertools

from Classifier import Classifier

#The main training function. For each fold, it loops through all possible combinations of
#the parameters listed in params, before generating a model for each combination and calling
#each model's train function.
def train(general_params):

    #Read in the data csv
    data_df = pd.read_csv(general_params['data_csv']) 

    #Instantiate classifier object
    classifier = Classifier()

    #List all possible parameters
    params = {'deep'  : [0,1],
              'bnorm' : [0,1],
              'drop'  : [0,1],
              'skip'  : [0,1]}

    #Iterate through all folds
    for fold_num in range(general_params['num_folds']):

        #Create catch-all directory for current fold
        fold_dir = general_params['save_path']+"\\Fold_"+str(fold_num)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        #Put all fold parameters in a dict
        fold_params = {"fold_dir" : fold_dir,
                       "fold_num" : fold_num,
                       "data_df"  : data_df}

        #Perform all network tests without and with augmented data
        for aug in [0,1]:        

            #Refresh the parameter generator
            param_gen = (dict(zip(params, x)) for x in itertools.product(*params.values()))

            #Read in the data according to the augmentation status
            classifier.load_fold_data(fold_num, data_df, aug)

            #Iterate through each possible parameter set in the param generator
            for model_params in param_gen:

                #Train the classifier
                classifier.train(fold_params, model_params)


def main():

    #Parse flags
    data_csv = FLAGS.data_csv
    output_folder = FLAGS.save_path
    num_folds = int(FLAGS.num_folds)

    #Save key params in a dict
    general_params = {"save_path"     : output_folder,
                      "num_folds"     : num_folds,
                      "data_csv"      : data_csv}

    #Call the train function
    train(general_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_csv',
        type=str,
        default="C:\\repos\\Courses\\ELEC874_FinalProject\\Code\\ELEC874_FullDataset.csv",
        help='Path to the csv file containing locations for all data used in training'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="C:\\repos\\Courses\\ELEC874_FinalProject\\NetworkOutputs\\All_Combinations",
        help='Path to the directory that contains the trained networks for each fold.'
    )   
    parser.add_argument(
        '--num_folds',
        type=str,
        default='4',
        help='Number of folds used in the K fold cross validation scheme.'
    )  

    #Get input flag arguments
    FLAGS, unparsed = parser.parse_known_args()

    main()