'''
The purpose of this file is to contain all helper functions
in one location for easy importing and cleaner code elsewhere.
'''

import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow
import tensorflow.keras
from tensorflow.keras import backend as K

def save_training_info(saveLocation, foldNum, params, trainingHistory, results, networkType, confusion_matrix=[]):

    metrics = params['metrics']

    numEpochsInt = len(trainingHistory["loss"])

    LinesToWrite = []
    LinesToWrite.append("Fold " +                str(foldNum) +"/"+ str(foldNum))
    LinesToWrite.append("\nNetwork type: " +     str(networkType))
    LinesToWrite.append("\nNumber of Epochs: " + str(numEpochsInt))
    LinesToWrite.append("\nBatch size: " +       str(params['batch_size']))
    LinesToWrite.append("\nLoss function: " +    str(params['loss_fxn']))

    LinesToWrite.append("\n\nTraining Statistics: ")
    LinesToWrite.append("\n\tFinal training loss: " + str(trainingHistory["loss"][numEpochsInt-1]))

    for i in range(len(metrics)):
        LinesToWrite.append("\n\tFinal training " + metrics[i] + ": " + str(trainingHistory[metrics[i]][numEpochsInt-1]))

    LinesToWrite.append("\n\tFinal validation loss: " + str(trainingHistory["val_loss"][numEpochsInt - 1]))

    for i in range(len(metrics)):
        LinesToWrite.append("\n\tFinal validation " + metrics[i] + ": " + str(trainingHistory["val_"+metrics[i]][numEpochsInt-1]))
        
    LinesToWrite.append("\n\nTesting Statistics: ")
    LinesToWrite.append("\n\tTest loss: " + str(results[0]))

    for i in range(len(metrics)):
        LinesToWrite.append("\n\tTest " + str(metrics[i]) + ": " + str(results[i+1]))

    if len(confusion_matrix) != 0:
        LinesToWrite.append("\n\n" + str(confusion_matrix))

    with open(os.path.join(saveLocation,"trainingInfo_"+networkType+".txt"),'w') as f:
        f.writelines(LinesToWrite)

def save_training_plot(saveLocation, history, metric, networkType):
    fig = plt.figure()
    numEpochs =len(history[metric])

    plt.plot([x for x in range(numEpochs)], history[metric], 'bo', label='Training '+metric)
    plt.plot([x for x in range(numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
    plt.title(networkType+' Training and Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(saveLocation, networkType+'_'+metric + '.png'))
    plt.close(fig)