import numpy as np
import pandas as pd 
import os, sys

#Directories to use
data_dir = "C:\\Datasets\\ELEC874\\FullDataset\\"
output_dir = "C:\\repos\\Courses\\ELEC874_FinalProject\\Code\\"

#Get all sequence ids in data directory
all_seq_ids = [f.name for f in os.scandir(data_dir) if f.is_dir()]

#Seperate out into different participants
ms01_ids = [x for x in all_seq_ids if x.startswith("MS01")]
ms02_ids = [x for x in all_seq_ids if x.startswith("MS02")]
ms03_ids = [x for x in all_seq_ids if x.startswith("MS03")]
ms04_ids = [x for x in all_seq_ids if x.startswith("MS04")]

ex01_ids = [x for x in all_seq_ids if x.startswith("AN01")]
ex02_ids = [x for x in all_seq_ids if x.startswith("AN02")]
ex03_ids = [x for x in all_seq_ids if x.startswith("AN03")]
ex04_ids = [x for x in all_seq_ids if x.startswith("NP01")]

#Seperate out into different folds
#Fold 0: Test set == MS04
#Fold 0: Test set == MS03
#Fold 0: Test set == MS02
#Fold 0: Test set == MS01

fold_0_ids = {'Train' : ms01_ids[:4] + ms02_ids[:4] + ms03_ids[:4] + ex01_ids[:4] + ex02_ids[:4] + ex03_ids[:4],
              'Validation'   : [ms01_ids[4],  ms02_ids[4],   ms03_ids[4], ex01_ids[4], ex02_ids[4], ex03_ids[4]],
              'Test'  : ms04_ids + ex04_ids}
fold_1_ids = {'Train' : ms01_ids[:4] + ms02_ids[:4] + ms04_ids[:4] + ex01_ids[:4] + ex02_ids[:4] + ex04_ids[:4],
              'Validation'   : [ms01_ids[4],  ms02_ids[4],   ms04_ids[4], ex01_ids[4], ex02_ids[4], ex04_ids[4]],
              'Test'  : ms03_ids + ex03_ids}
fold_2_ids = {'Train' : ms01_ids[:4] + ms03_ids[:4] + ms04_ids[:4] + ex01_ids[:4] + ex03_ids[:4] + ex04_ids[:4],
              'Validation'   : [ms01_ids[4],  ms03_ids[4],   ms04_ids[4], ex01_ids[4], ex03_ids[4], ex04_ids[4]],
              'Test'  : ms02_ids + ex02_ids}
fold_3_ids = {'Train' : ms02_ids[:4] + ms03_ids[:4] + ms04_ids[:4] + ex02_ids[:4] + ex03_ids[:4] + ex04_ids[:4],
              'Validation'   : [ms02_ids[4],  ms03_ids[4],   ms04_ids[4], ex02_ids[4], ex03_ids[4], ex04_ids[4]],
              'Test'  : ms01_ids + ex01_ids}
all_folds = [fold_0_ids, fold_1_ids, fold_2_ids, fold_3_ids]

'''
fold_0_ids = {'Train' : [ms01_ids[1], ms01_ids[2], ms01_ids[3]],
              'Validation'   : [ms01_ids[4]],
              'Test'  : [ms01_ids[0]]}
all_folds = [fold_0_ids]
'''


#Initialize a blank dataframe
dataset_df = pd.DataFrame(index=[], columns=['Fold', 'Set', 'Folder', 'US_Image', 'US_Pose'])

#For each fold
for fold_num, fold_dict in enumerate(all_folds):

    #For each of the Training / Validation / Testing sections
    for section, id_list in fold_dict.items():

        #For each ID within the current section
        for seq_id in id_list:

            #Save the current directory with all entries from this id
            print(seq_id)
            curr_dir = data_dir + seq_id

            #Open the csv for this id
            curr_csv = curr_dir + "\\Ultrasound\\" + seq_id + "_Ultrasound_Labels.csv"
            curr_df = pd.read_csv(curr_csv)

            #Rename the us image and probe orientation columns
            curr_df.rename(columns= {'FileName':'US_Image', 'Probe_Orientation' : 'US_Pose'}, inplace=True)

            #Remove the Time Recorded column
            del curr_df['Unnamed: 0']
            del curr_df['Time Recorded']
            del curr_df['VesselsCombined']

            #Add the fold, set and directory as columns in this dataframe
            curr_df['Fold']   = str(fold_num)
            curr_df['Set']    = str(section)
            curr_df['Folder'] = "FullDataset\\" + seq_id

            dataset_df = pd.concat([dataset_df, curr_df], ignore_index=True)

us_poses = np.array(dataset_df['US_Pose'].unique())
print("All US Poses before: " + str(us_poses))

dataset_df['US_Pose'] = dataset_df['US_Pose'].fillna('Undefined')

us_tasks = np.array(dataset_df['US_Pose'].unique())
print("All US poses after: " + str(us_tasks))

#Write the resulting df to a csv
dataset_df.to_csv(output_dir + "ELEC874_FullDataset.csv")