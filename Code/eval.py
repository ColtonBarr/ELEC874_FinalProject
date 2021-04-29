'''
This file encapsulates the evaluation script for 
all the models.
'''

import numpy as np
import pandas as pd
import argparse
import HelperFxns as fxns
import os, sys
import itertools
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols

from Classifier import Classifier

#Taken from the Classifier class; used to generate a model's name string from parameters.
def generate_model_name(param_set_dict):
    flag_strs = list(param_set_dict.keys())
    flag_bools = list(param_set_dict.values())

    suffix = '_'.join(tuple([flag_strs[i] for i in range(5) if flag_bools[i]]))

    return "CNN_" + suffix

#The core evaluation function. It creates each model and calls the .evaluate() method,
#followed by processing and writing the results to collector lists for eventual generation
#of an output .txt file.
def evaluate(general_params):

    #Initialize a list to append results to
    report = []
    report.append(" --------------------------------------------- ")
    report.append("\n ------- Individual Classifier Results ------- ")
    report.append("\n --------------------------------------------- \n")

    #Read in the data csv
    data_df = pd.read_csv(general_params['data_csv']) 

    #Instantiate classifier object
    classifier = Classifier()

    #List all possible parameters
    params = {'aug'   : [0,1],
              'deep'  : [0,1],
              'bnorm' : [0,1],
              'drop'  : [0,1],
              'skip'  : [0,1]}

    results_dicts = []

    #Iterate through all folds
    for fold_num in range(general_params['num_folds']):

        report.append("\n ~~~ Fold 0 ~~~ \n\n")

        #Read in the data according to the augmentation status
        classifier.load_fold_data(fold_num, data_df)

        #Determine the directory for the current fold
        fold_dir = general_params['model_path']+"\\Fold_"+str(fold_num)

        #Generate all possible combinations of all the params
        param_gen = (dict(zip(params, x)) for x in itertools.product(*params.values()))

        #Iterate through each model
        for param_set in param_gen:

            #Load the model with the corresponding parameters
            model_name = generate_model_name(param_set)
            full_model_path = os.path.join(fold_dir, model_name, model_name)            
            model = classifier.load_model(full_model_path)

            #Get overall performance information
            accuracy, confusion_matrix, overall_report = classifier.evaluate(model)

            #Append information to the report
            report.append("---------------------------------------------------------------------\n")
            report.append(" --- Classifier: " + model_name + " --- \n\n")
            report.append(str(overall_report))
            report.append("\n Confusion Matrix: \n\n")
            report.append(str(confusion_matrix) + "\n")
            report.append("\n---------------------------------------------------------------------\n")    

            #Add the accuracy to the param dict and append to a collector list
            param_set['accuracy'] = accuracy
            results_dicts.append(param_set)

    #Initialize a new list for the ANOVA results
    anova_report = []

    anova_report.append(" --------------------------------------------- ")
    anova_report.append("\n --------------- ANOVA Results --------------- ")
    anova_report.append("\n --------------------------------------------- \n")

    #Generate a dataframe from the list of dicts
    results_df = pd.DataFrame(results_dicts)

    #Interate through all independent variables
    for var in params.keys():

        #Perform a 1 way ANOVA for each parameter
        model = ols('accuracy ~ C(' + var + ')', data=results_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(" ~~~ " + var + " ~~~ ")
        print(anova_table)
        
        #Append information to the results list
        anova_report.append(" ~~~ " + var + " ~~~ \n\n")
        anova_report.append(str(anova_table) + "\n")
        anova_report.append("\n\n") 


    #Insert the anova report at the start of the overall report
    for ln in reversed(anova_report):

        report.insert(0, ln)

    #Determine the classifier with the highest accuracy
    sorted_df = results_df.sort_values(by=['accuracy'], ascending=False)
      
    #Add all the classifiers to another list
    top_report = []
    top_report.append(" --------------------------------------------- ")
    top_report.append("\n ---------- Top Performing Networks ---------- ")
    top_report.append("\n --------------------------------------------- \n\n")    

    #Iterate through all classifiers
    for i in range(32):

        #Get the ith model name and accuracy
        ith_dict = dict(sorted_df.iloc[i])
        model_name = generate_model_name(ith_dict)

        top_report.append(" #" + str(i+1) + " Network: " + model_name + "\n")
        top_report.append(" Accuracy: " + str(ith_dict['accuracy']) + "\n\n")

    #Insert the topreport at the start of the overall report
    for ln in reversed(top_report):
        report.insert(0, ln)

    #Write the results to the evaluation report file.
    with open(os.path.join(general_params['report_path'], "EvaluationReport.txt"),'w') as f:
        f.writelines(report)



def main():

    #Parse flags
    data_csv = FLAGS.data_csv
    model_folder = FLAGS.model_path
    num_folds = int(FLAGS.num_folds)
    report_path = FLAGS.report_path

    #Generate output dict
    general_params = {"model_path"     : model_folder,
                      "num_folds"     : num_folds,
                      "data_csv"      : data_csv,
                      "report_path"   : report_path}

    #Call evaluate function
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
        default="C:\\repos\\Courses\\ELEC874_FinalProject\\NetworkOutputs\\All_Combinations",
        help='Path to the directory that contains the trained networks for each fold.'
    )   
    parser.add_argument(
        '--num_folds',
        type=str,
        default='1',
        help='Number of folds used in the K fold cross validation scheme.'
    )  
    parser.add_argument(
        '--report_path',
        type=str,
        default='C:\\repos\\Courses\\ELEC874_FinalProject',
        help='The directory to save the evaluation report in.'
    )  

    #Get input flag arguments
    FLAGS, unparsed = parser.parse_known_args()

    main()