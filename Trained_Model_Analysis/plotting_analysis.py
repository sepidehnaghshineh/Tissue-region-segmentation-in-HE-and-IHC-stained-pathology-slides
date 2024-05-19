import pandas as pd
import os
import sys
sys.path.append('./')
import ast
import csv
import re
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tools.data_visualising import  plt_charts, plot_roc_and_metrics , plot_confusion_matrix_and_save , save_wrongly_predicted_patches, calculate_confidence_intervals , process_results

parser = argparse.ArgumentParser()

# Arguments to be provided before running the code.

parser.add_argument('--data_source', default='source_name', help='TCGA or Camelyon16 or Camelyon17 or HER2 or HER2OHE or ...', dest='data_source')
parser.add_argument('--ext_data_source', default='_', help='ext_HER2_HE or Camelyon17 or HER2 or HER2OHE or BAU', dest='ext_data_source')
parser.add_argument('--Level_patch', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of steps of execution', dest='num_epochs')
parser.add_argument('--model_type', default='LeNet5', help='mycnn1 or mycnn2 or mycnn3 or ...', dest='model_type')
parser.add_argument('--device_type', default='cuda', help='cuda or cpu', dest='device_type')
parser.add_argument('--result_type_val', default='val', help='val', dest='result_type_val')
parser.add_argument('--result_type_test', default='test', help='test', dest='result_type_test')
parser.add_argument('--result_type_external', default='external_val', help='external_val', dest='result_type_external')


FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)




# Create the directory to save the results
if not os.path.exists('./results'):
    os.mkdir('./results')   
if not os.path.exists('./results/{}'.format(FLAGS.data_source)):
    os.mkdir('./results/{}'.format(FLAGS.data_source))
if not os.path.exists('./results/{}/{}/'.format(FLAGS.data_source, FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}/'.format(FLAGS.data_source, FLAGS.Level_patch))
if not os.path.exists('./results/{}/{}/'.format(FLAGS.data_source, FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}/'.format(FLAGS.data_source, FLAGS.Level_patch))
if not os.path.exists('./results/{}/{}/train_test_summary'.format(FLAGS.data_source, FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}/train_test_summary'.format(FLAGS.data_source, FLAGS.Level_patch))
if not os.path.exists('./results/{}/{}/train_test_summary/{}'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)):
    os.mkdir('./results/{}/{}/train_test_summary/{}'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type))

################################################################
class_label_mapping = {'background': 0, 'tissue': 1}
class_names = ['background', 'tissue']
background_classes = ['background']
tissue_classes = ['tissue']

########################## plot charts for accuracy and loss for train and validation : ########################################

train_data = pd.read_csv('./results/{}/{}/train_test_summary/{}/train_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))
val_data = pd.read_csv('./results/{}/{}/train_test_summary/{}/val_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))
val_results = pd.read_csv('./results/{}/{}/train_test_summary/{}/validation_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))
plt_charts(train_data['loss'], train_data['accuracy'], val_data['loss'], val_data['accuracy'], FLAGS.num_epochs, './results/{}/{}/train_test_summary/{}'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type),FLAGS.data_source, FLAGS.Level_patch)

################### validation ################
y_true_numeric_val, y_prob_raw_val, y_pred_val = process_results(val_results, class_label_mapping)
auc_val, f1_score_val, recall_score_val, precision_val = plot_roc_and_metrics(val_results, class_label_mapping, FLAGS.result_type_val, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
plot_confusion_matrix_and_save(val_results, class_names, FLAGS.result_type_val, FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
save_wrongly_predicted_patches(val_results, FLAGS.result_type_val,FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
calculate_confidence_intervals(y_true_numeric_val, y_prob_raw_val, FLAGS.result_type_val, FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)

######################## TEST ####################

test_data = pd.read_csv('./results/{}/{}/train_test_summary/{}/test_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))
test_results = pd.read_csv('./results/{}/{}/train_test_summary/{}/test_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))


y_true_numeric_test, y_prob_raw_test, y_pred_test = process_results(test_results, class_label_mapping)
auc_test, f1_score_test, recall_score_test , precision_test = plot_roc_and_metrics(test_results, class_label_mapping, FLAGS.result_type_test, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
plot_confusion_matrix_and_save(test_results, class_names, FLAGS.result_type_test, FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
save_wrongly_predicted_patches(test_results, FLAGS.result_type_test, FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type)
calculate_confidence_intervals(y_true_numeric_test, y_prob_raw_test, FLAGS.result_type_test, FLAGS.ext_data_source, FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)


###################################################   RESULTS TABLE ##############################################################

#Create an empty DataFrame to store the results

Train_Val_Dataset_summary= pd.read_csv('./results/{}/{}/train_test_summary/{}/Train_Val_Dataset_summary.txt'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type))
Test_Dataset_summary = pd.read_csv('./results/{}/{}/train_test_summary/{}/Test_Dataset_summary.txt'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type))

best_epoch = Train_Val_Dataset_summary['Best_Epoch'].iloc[0]

# Parse the lists inside Train_Label counts and Val_Label counts
Train_Val_Dataset_summary['Train_Label_counts'] = Train_Val_Dataset_summary['Train_Label_counts'].apply(ast.literal_eval)
Train_Val_Dataset_summary['Val_Label_counts'] = Train_Val_Dataset_summary['Val_Label_counts'].apply(ast.literal_eval)
Test_Dataset_summary['Test_Label_counts'] = Test_Dataset_summary['Test_Label_counts'].apply(ast.literal_eval)

# Extract the number of total patches
train_total_patchs = Train_Val_Dataset_summary['Train_total_patchs'].iloc[0]
# print('Train_total_patchs = {}'.format(train_total_patchs))

val_total_patchs = Train_Val_Dataset_summary['Val_total_patchs'].iloc[0]
# print('Val_total_patchs = {}'.format(val_total_patchs))

test_total_patchs = Test_Dataset_summary['Test_total_patchs'].iloc[0]
# print('Test_total_patchs = {}'.format(test_total_patchs))

########################

# Extract the counts for "background" and "tissue" labels
train_bg_count = Train_Val_Dataset_summary['Train_Label_counts'].iloc[0][0]
train_tissue_count = Train_Val_Dataset_summary['Train_Label_counts'].iloc[0][1]

val_bg_count = Train_Val_Dataset_summary['Val_Label_counts'].iloc[0][0]
val_tissue_count = Train_Val_Dataset_summary['Val_Label_counts'].iloc[0][1]

test_bg_count = Test_Dataset_summary['Test_Label_counts'].iloc[0][0]
test_tissue_count = Test_Dataset_summary['Test_Label_counts'].iloc[0][1]



##################
# Filter rows for the best epoch in train_data
best_epoch_data_train = train_data[train_data.iloc[:, 0] == best_epoch]

# Filter rows for the best epoch in val_data
best_epoch_data_val = val_data[val_data.iloc[:, 0] == best_epoch]

# Extract loss and accuracy for the best epoch from train_data and val_data
best_epoch_loss_train = best_epoch_data_train.iloc[:, 1].values[0]
best_epoch_accuracy_train = best_epoch_data_train.iloc[:, 2].values[0]

best_epoch_loss_val = best_epoch_data_val.iloc[:, 1].values[0]
best_epoch_accuracy_val = best_epoch_data_val.iloc[:, 2].values[0]

# Extract loss and accuracy values from test_data
test_loss = test_data.iloc[0, 1]
test_accuracy = test_data.iloc[0, 2]

# File paths
experiments_results_file = './results/{}experiments_results.csv'.format(FLAGS.data_source)
all_datasets_summary_file = './results/{}/All_Datasets_summary.csv'.format(FLAGS.data_source)

# Check if the CSV files exist
if not os.path.exists(all_datasets_summary_file):
    # Create the experiments_results CSV with column headers
    All_Datasets_summary = pd.DataFrame({'Data_Source': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                        'train_total': [train_total_patchs],
                                        'train_tissue': [train_tissue_count],
                                        'train_bg': [train_bg_count],
                                        'val_total': [val_total_patchs],
                                        'val_tissue': [val_tissue_count],
                                        'val_bg': [val_bg_count],
                                        'test_total':[test_total_patchs],
                                        'test_tissue': [test_tissue_count],
                                        'test_bg': [test_bg_count]},
                                       columns=['Data_Source', 'train_total', 'train_tissue', 'train_bg', 'val_total', 'val_tissue', 'val_bg', 'test_total','test_tissue', 'test_bg'])

    # Save the CSV file
    All_Datasets_summary.to_csv(all_datasets_summary_file, index=False)

if not os.path.exists(experiments_results_file):
    # Create the All_Datasets_summary CSV with column headers
    Experiments_results = pd.DataFrame({'Data_Source': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                        'train_loss': [best_epoch_loss_train],
                                        'train_accuracy': [best_epoch_accuracy_train],
                                        'val_loss': [best_epoch_loss_val],
                                        'val_accuracy': [best_epoch_accuracy_val],
                                        'test_loss': [test_data['loss'].iloc[0]],  # Use iloc[0] to get the value
                                        'test_accuracy': [test_data['accuracy'].iloc[0]],  # Use iloc[0] to get the value
                                        'val_AUC': [auc_val],
                                        'test_AUC': [auc_test],
                                        'val_F1score': [f1_score_val],
                                        'test_F1score': [f1_score_test],
                                        'recall_score_validation': [recall_score_val],
                                        'recall_score_test': [recall_score_test],
                                        'precision_val': [precision_val],
                                        'precision_test': [precision_test]
                                        },
                                    
                                       columns=['Data_Source', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy', 'val_AUC', 'test_AUC', 'val_F1score', 'test_F1score', 'recall_score_validation', 'recall_score_test', 'precision_val', 'precision_test'])

    # Save the CSV file
    Experiments_results.to_csv(experiments_results_file, index=False)

# Append new results to the CSV files
if os.path.exists(experiments_results_file) and os.path.exists(all_datasets_summary_file):
    # Load the existing data from the CSV files
    existing_experiments_results = pd.read_csv(experiments_results_file)
    existing_all_datasets_summary = pd.read_csv(all_datasets_summary_file)

    # Check if the new data is a duplicate
    data_source = FLAGS.data_source +'_' + FLAGS.Level_patch
    is_duplicate_experiments = existing_experiments_results['Data_Source'] == data_source
    is_duplicate_all_datasets = existing_all_datasets_summary['Data_Source'] == data_source

    if not is_duplicate_all_datasets.all():
        # Append new results to the experiments_results CSV
        new_row_all_datasets = pd.DataFrame({'Data_Source': [data_source],
                                            'train_total': [train_total_patchs],
                                            'train_tissue': [train_tissue_count],
                                            'train_bg': [train_bg_count],
                                            'val_total': [val_total_patchs],
                                            'val_tissue': [val_tissue_count],
                                            'val_bg': [val_bg_count],
                                            'test_total' : [test_total_patchs],
                                            'test_tissue': [test_tissue_count],
                                            'test_bg': [test_bg_count]})

        updated_all_datasets_summary = pd.concat([existing_all_datasets_summary, new_row_all_datasets], ignore_index=True)
        updated_all_datasets_summary.to_csv(all_datasets_summary_file, index=False)

    if not is_duplicate_experiments.all():
        # Append new results to the All_Datasets_summary CSV
        new_row_experiments = pd.DataFrame({'Data_Source': [data_source],
                                            'train_loss': [best_epoch_loss_train],
                                            'train_accuracy': [best_epoch_accuracy_train],
                                            'val_loss': [best_epoch_loss_val],
                                            'val_accuracy': [best_epoch_accuracy_val],
                                            'test_loss': [test_data['loss'].iloc[0]],  # Use iloc[0] to get the value
                                            'test_accuracy': [test_data['accuracy'].iloc[0]],  # Use iloc[0] to get the value
                                            'val_AUC': [auc_val],
                                            'test_AUC': [auc_test],
                                            'val_F1score': [f1_score_val],
                                            'test_F1score': [f1_score_test],
                                            'recall_score_validation': [recall_score_val],
                                            'recall_score_test': [recall_score_test],
                                            'precision_val': [precision_val],
                                            'precision_test': [precision_test]})

        updated_experiments_results = pd.concat([existing_experiments_results, new_row_experiments], ignore_index=True)
        updated_experiments_results.to_csv(experiments_results_file, index=False)

