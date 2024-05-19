import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import matplotlib.colors as mcolors
import argparse
parser = argparse.ArgumentParser(
    description='Statistical Test ')

parser.add_argument('--dataset_dir', default='./Dataset_Prep/data_preparation_results/Cropped_slides/source_name/test/allpatches__level_4_size_128_.txt',help='testing dataset path', dest='dataset_dir')   
parser.add_argument('--data_source', default='source_name', help='TCGA or Camelyon17 or HER2 or HER2OHE or BAU', dest='data_source')
parser.add_argument('--Level_patch_1', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch_1')
parser.add_argument('--Level_patch_2', default='L6_32', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch_2')
parser.add_argument('--Level_patch_3', default='L6_32', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch_3')

parser.add_argument('--ext_data_source', default='_', help='ext_HER2_HE or Camelyon17 or HER2 or HER2OHE or BAU', dest='ext_data_source')
parser.add_argument('--model_type', default='LeNet5',help='the used CNN model', dest='model_type')
parser.add_argument('--model_mode', default='test', help='model_mode', dest='model_mode')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)






directories_to_create = [
    './results',
    './results/{}'.format(FLAGS.data_source),
    './results/{}/{}'.format(FLAGS.data_source, FLAGS.num_type_test),
    './results/{}/{}/MC_Nemartest_analysis'.format(FLAGS.data_source, FLAGS.num_type_test),
    './results/{}/{}/MC_Nemartest_analysis/{}'.format(FLAGS.data_source, FLAGS.num_type_test, FLAGS.Level_patch_1 + '_' + FLAGS.Level_patch_2)
]

for directory in directories_to_create:
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass  # If the directory already exists, do nothing

#################################################################################################################################
"""
Merging the two CSV files
 to make patchs in the same order for both levels and patch sizes 
 to make it easy to compare for MC Nemar test
 """ 

# Read the test results CSV files for both models
test_results_model_1 = pd.read_csv('./results/{}/{}/{}/train_test_summary/{}/{}/{}/test_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch_1, FLAGS.run_num,FLAGS.model_type))  # Replace with your file path
test_results_model_2 = pd.read_csv('./results/{}/{}/{}/train_test_summary/{}/{}/{}/test_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch_2, FLAGS.run_num,FLAGS.model_type))  # Replace with your file path

# Extract last two parts of the patch path
test_results_model_1['patch_key_model1'] = test_results_model_1['patch_path'].str.split('/').str[-3:].apply('/'.join)
test_results_model_2['patch_key_model2'] = test_results_model_2['patch_path'].str.split('/').str[-3:].apply('/'.join)

# Merge based on the extracted keys
merged_results = pd.merge(test_results_model_1, test_results_model_2, left_on='patch_key_model1', right_on='patch_key_model2')


# Filter out rows where the patch paths from both models are the same
filtered_results = merged_results[merged_results['patch_path_x'] != merged_results['patch_path_y']]

# Select columns of interest
filtered_results = filtered_results[['patch_path_x', 'patch_path_y', 'y_true_x', 'y_pred_x','y_prob_raw_x', 'y_true_y', 'y_pred_y','y_prob_raw_y' ]]

# Rename columns for clarity
filtered_results.columns = ['patch_path_model1', 'patch_path_model2', 'y_true_model1', 'y_pred_model1','y_prob_raw_model1', 'y_true_model2', 'y_pred_model2', 'y_prob_raw_model2']

# Save the filtered results to a CSV file
filtered_results.to_csv('./results/{}/{}/MC_Nemartest_analysis/{}/filtered_results.csv'.format(FLAGS.data_source, FLAGS.num_type_test, FLAGS.Level_patch_1 + '_vs_' + FLAGS.Level_patch_2 ), index=False)

########################################################################################################################################
""" MC Nemar test"""
# Read the CSV file
file_path = './results/{}/MC_Nemartest_analysis/{}/filtered_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch_1 + '_vs_' + FLAGS.Level_patch_2 )
data = pd.read_csv(file_path)

# Extract necessary columns
y_true_model1 = data['y_true_model1']
y_pred_model1 = data['y_pred_model1']
y_true_model2 = data['y_true_model2']
y_pred_model2 = data['y_pred_model2']

# Calculate correct predictions for each model
correct_preds1 = (y_true_model1 == y_pred_model1)
correct_preds2 = (y_true_model2 == y_pred_model2)

# Create contingency table
contingency_table = np.zeros((2, 2), dtype=int)
correct_correct = contingency_table[0, 0] = np.sum(correct_preds1 & correct_preds2)
correct_wrong = contingency_table[0, 1] = np.sum(correct_preds1 & ~correct_preds2)
wrong_correct = contingency_table[1, 0] = np.sum(~correct_preds1 & correct_preds2)
wrong_wrong = contingency_table[1, 1] = np.sum(~correct_preds1 & ~correct_preds2)

# Perform McNemar test
result = mcnemar(contingency_table, exact=True)

print(f'Correct-Correct: {correct_correct}')
print(f'Correct-Wrong: {correct_wrong}')
print(f'Wrong-Correct: {wrong_correct}')
print(f'Wrong-Wrong: {wrong_wrong}')
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
p_values_corrected = multipletests([result.pvalue], alpha=alpha, method='bonferroni')[1]
if p_values_corrected[0] > alpha:
    test_result = 'Same proportions of errors (fail to reject H0)'
    print('Same proportions of errors (fail to reject H0)')
else:
    test_result = 'Different proportions of errors (reject H0)'
    print('Different proportions of errors (reject H0)')


# Create a DataFrame for the values
values_dict = {
    'model1_model2': [FLAGS.data_source + '_' + FLAGS.Level_patch_1 + '__vs__' + FLAGS.data_source + '_' + FLAGS.Level_patch_2],
    'Correct-Correct': [correct_correct],
    'Correct-Wrong': [correct_wrong],
    'Wrong-Correct': [wrong_correct],
    'Wrong-Wrong': [wrong_wrong],
    'statistic': [result.statistic],
    'p-value': [result.pvalue]
}

values_df = pd.DataFrame(values_dict)

# Save values to a CSV file
values_df.to_csv('./results/{}/MC_Nemartest_analysis/{}/contingency_MC_Nemar_values.csv'.format(FLAGS.data_source, FLAGS.Level_patch_1 + '_vs_' + FLAGS.Level_patch_2), index=False)


# Create a DataFrame for wrong_wrong cases
wrong_wrong_df = filtered_results[(filtered_results['y_true_model1'] != filtered_results['y_pred_model1']) & (filtered_results['y_true_model2'] != filtered_results['y_pred_model2'])]

# Select columns of interest for the wrong_wrong condition
wrong_wrong_data = wrong_wrong_df[['patch_path_model1', 'patch_path_model2', 'y_true_model1', 'y_pred_model1','y_prob_raw_model1', 'y_true_model2', 'y_pred_model2', 'y_prob_raw_model2']]

# Save the wrong_wrong data to a CSV file
wrong_wrong_data.to_csv('./results/{}/MC_Nemartest_analysis/{}/wrong_wrong_patches.csv'.format(FLAGS.data_source, FLAGS.Level_patch_1 + '_vs_' + FLAGS.Level_patch_2), index=False, header=['patch_path_model1', 'patch_path_model2', 'y_true_model1', 'y_pred_model1','y_prob_raw_model1', 'y_true_model2', 'y_pred_model2', 'y_prob_raw_model2'])

######################################################################################################################

""" Plotting the MC Nemar test """
new_cmap = mcolors.ListedColormap(['#FFEDA0', '#9A8C98', '#38686A', '#007991'])

# Labels for each cell
cell_labels = [
    [f'Correct_Correct: {contingency_table[0, 0]}', f'Correct_Wrong: {contingency_table[0, 1]}'],
    [f'Wrong_Correct: {contingency_table[1, 0]}', f'Wrong_Wrong: {contingency_table[1, 1]}']
]
# Create the heatmap
plt.figure(figsize=(10, 10))
plt.imshow(contingency_table, cmap= new_cmap, interpolation='nearest')

# Show cell values
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'\n{cell_labels[i][j]}\n', ha='center', va='center', fontsize=15, fontweight='bold', color='black')
        # plt.text(j, i, cell_labels[i][j], horizontalalignment='center', verticalalignment='center', color='black')

# Add labels for axes
# plt.xlabel('Model 2 Predictions')
# plt.ylabel('Model 1 Predictions')
plt.xlabel('Model 2 : ' +(FLAGS.data_source + '_' + FLAGS.Level_patch_2 ) + '  Predictions')
plt.ylabel('Model 1 : ' +(FLAGS.data_source + '_' + FLAGS.Level_patch_1 ) + '  Predictions')
# Add a colorbar
plt.colorbar(label='Count')

# Title
plt.title('MC Nemar test Contingency Table'+ '_' + test_result)
# Specify the directory to save the plot
plot_directory = './results/{}/MC_Nemartest_analysis/{}/McNemar_Test_plot.pdf'.format(FLAGS.data_source, FLAGS.Level_patch_1 + '_vs_' + FLAGS.Level_patch_2)
# Save the plot as an image file
plt.savefig(plot_directory)

# #############################################################################################################
