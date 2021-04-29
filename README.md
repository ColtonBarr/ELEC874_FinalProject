# ELEC874_FinalProject
This repo contains the full implementation of my ELEC 874 Final Project. 

## Requirements

This project uses [Conda](https://docs.conda.io/en/latest/) for package management. Assuming Conda has been installed and added to path, the environment can be setup using:

```setup
conda env create -f environment.yml
```

The environment can then be activated using:

```activate
conda activate ELEC_Final_Project
```

Ensure that you activate the environment and change your working directory to the *Code* folder within this cloned repo to run the following commands.

## Dataset

Since the full dataset has a ~30hr training time, a sample dataset has been created to demonstrate this project's full training and evaluation pipeline. The sample dataset can be downloaded [here](https://queensuca-my.sharepoint.com/:u:/g/personal/14cb49_queensu_ca/ETyJdufwIXpMrEVXQ52qGpEB2-2g6XskbN0lzk5TvSQc8Q?e=SFPqAw).

Download and extract the folder *SampleDataset* to the cloned folder for this repository, such that the parent folder for *LabelledData* is *ELEC874_FinalProject*.

## Training

To train the full set of 32 networks, run this single command in the ELEC_Final_Project environment:

```command1
python train.py --data_csv <path_to_sample_dataset_csv> --save_path <empty_directory_for_results> --num_folds 1
```

Note that rich information about each trained network, including test results and metric graphs are available in the individual output folders for each model.

## Evaluation

This script generates a .txt report that includes the following information:
1. Ranked list of all networks and their respective classification accuracies
2. Full results from the one-way ANOVA test for each of the 5 architectural modifications.
3. Full test report for each classifier including precision, recall, accuracy and confusion matrix.

To create this report for the networks trained using the ```train.py``` script, run the following command in the ELEC_Final_Project environment:

```command2
python eval.py --data_csv <path_to_sample_dataset_csv> --model_path <save_path_from_training> --num_folds 1 --report_path <directory_to_save_report_in>
```

## Pre-trained Models

The 32 trained models used to generate the final results in this paper are available in the folder *AllTrainedNetworks*. To generate the full evaluation report for these networks using the sample data, run the following command in the ELEC_Final_Project environment:

```command3
python eval.py --data_csv <path_to_sample_dataset_csv> --model_path ..\AllTrainedNetworks --num_folds 1 --report_path <directory_to_save_report_in>
```

Note that the results generated here will differ from those reported in the paper since the sample dataset has a different test set.

## Results

The raw evaluation report for the full dataset, from which all results reported in the paper are drawn, is available as *FullEvaluationReport.txt* in this repo.
