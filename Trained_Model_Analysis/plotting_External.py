import ast
import sys
sys.path.append('./')
import argparse
import itertools
from tools.external_visualizing import  plt_charts, plot_roc_and_metrics , plot_confusion_matrix_and_save , save_wrongly_predicted_patches, calculate_confidence_intervals , process_results


parser = argparse.ArgumentParser()

# Arguments to be provided before running the code.

parser.add_argument('--data_source', default='source_name', help='TCGA or Camelyon16 or Camelyon17 or HER2 or HER2OHE or ...', dest='data_source')
parser.add_argument('--ext_data_source', default='External_for_Combined', help='ext_HER2_HE or Camelyon17 or HER2 or HER2OHE or BAU', dest='ext_data_source')
parser.add_argument('--Level_patch', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of steps of execution', dest='num_epochs')
parser.add_argument('--model_type', default='LeNet5', help='mycnn1 or mycnn2 or mycnn3 or ...', dest='model_type')
parser.add_argument('--train_type', default='just_Student', help='train type : just_Teacher or just_Student  or model_distillation', dest='train_type')
parser.add_argument('--device_type', default='cuda', help='cuda or cpu', dest='device_type')
parser.add_argument('--result_type_val', default='val', help='val', dest='result_type_val')
parser.add_argument('--result_type_test', default='test', help='test', dest='result_type_test')
parser.add_argument('--result_type_external', default='external_val', help='external_val', dest='result_type_external')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)



if not os.path.exists('./Tissue_region_Segmentation/results/external'):
    os.mkdir('./Tissue_region_Segmentation/results/external')

if not os.path.exists('./Tissue_region_Segmentation/results/external/{}'.format(FLAGS.data_source)):
    os.mkdir('./Tissue_region_Segmentation/results/external/{}'.format(FLAGS.data_source))

################################################################
class_label_mapping = {'background': 0, 'tissue': 1}
class_names = ['background', 'tissue']
background_classes = ['background']
tissue_classes = ['tissue']

################### Extra Validation #############
external_val_data = pd.read_csv('./Tissue_region_Segmentation/results/external/{}/{}_external_val_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch))
external_val_results =pd.read_csv('./Tissue_region_Segmentation/results/external/{}/{}_external_val_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch))

y_true_numeric_external, y_prob_raw_external, y_pred_external = process_results(external_val_results, class_label_mapping)
auc_external, f1_score_external, recall_score_external , precision_external= plot_roc_and_metrics(external_val_results, class_label_mapping, FLAGS.result_type_external, FLAGS.data_source)
plot_confusion_matrix_and_save(external_val_results, class_names,FLAGS.ext_data_source, FLAGS.data_source)
save_wrongly_predicted_patches(external_val_results,FLAGS.ext_data_source, FLAGS.data_source)

# ############# EXTERNAL VALIDATION RESULTS ##########################
External_val_Dataset_summary = pd.read_csv('./Tissue_region_Segmentation/results/external/{}/external_Dataset_summary.txt'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.run_num,FLAGS.model_type, FLAGS.train_type, FLAGS.num_type_test,  FLAGS.ext_data_source))
External_val_Dataset_summary['external_Label_counts'] = External_val_Dataset_summary['external_Label_counts'].apply(ast.literal_eval)
external_total_patchs = External_val_Dataset_summary['external_total_patchs'].iloc[0]
external_bg_count = External_val_Dataset_summary['external_Label_counts'].iloc[0][0]
external_tissue_count = External_val_Dataset_summary['external_Label_counts'].iloc[0][1]
# Extract loss and accuracy values from external validation_data
external_loss = external_val_data.iloc[0, 1]
external_accuracy = external_val_data.iloc[0, 2]
# File paths

All_external_validation_results_file = './Tissue_region_Segmentation/results/external/All_External_validation.csv'.format(FLAGS.data_source)
all_external_datasets_summary_file = './Tissue_region_Segmentation/results/external/All_External_Datasets_summary.csv'.format(FLAGS.data_source)


# Check if the CSV files exist
if not os.path.exists(all_external_datasets_summary_file):
    # Create the experiments_results CSV with column headers
    All_External_Datasets_summary = pd.DataFrame({'Model': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                         'External_Data_Source': [FLAGS.ext_data_source],
                                        'external_total':[external_total_patchs],
                                        'external_tissue': [external_tissue_count],
                                        'external_bg': [external_bg_count]},
                                       columns=['Model','External_Data_Source','external_total','external_tissue', 'external_bg'])

    # Save the CSV file
    All_External_Datasets_summary.to_csv(all_external_datasets_summary_file, index=False)

if not os.path.exists(All_external_validation_results_file):
    # Create the All_Datasets_summary CSV with column headers
    All_external_validation_results = pd.DataFrame({'Model': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                        'External_Data_Source': [FLAGS.ext_data_source],
                                        'external_loss': [external_loss],  # Use iloc[0] to get the value
                                        'external_accuracy': [external_accuracy],  # Use iloc[0] to get the value
                                        'external_AUC': [auc_external],
                                        'external_F1score': [f1_score_external],
                                        'recall_score_external': [recall_score_external]},

                                       columns=['Model', 'External_Data_Source', 'external_loss', 'external_accuracy', 'external_AUC', 'external_F1score', 'recall_score_external'])

    # Save the CSV file
    All_external_validation_results.to_csv(All_external_validation_results_file, index=False)
    

# #Append new results to the CSV files
if os.path.exists(All_external_validation_results_file) and os.path.exists(all_external_datasets_summary_file):
    # Load the existing data from the CSV files
    existing_external_validation_results = pd.read_csv(All_external_validation_results_file)
    existing_external_datasets_summary = pd.read_csv(all_external_datasets_summary_file)
    # print(existing_external_datasets_summary.columns)

    # Check if the new data is a duplicate
    external_data_source = FLAGS.ext_data_source +'_' + FLAGS.Level_patch
    is_duplicate_external_validation = existing_external_validation_results['External_Data_Source'] == external_data_source
    is_duplicate_external_datasets = existing_external_datasets_summary['External_Data_Source'] == external_data_source

    if not is_duplicate_external_datasets.all():
        # Append new results to the experiments_results CSV
        new_row_external_datasets = pd.DataFrame({'Model': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                         'External_Data_Source': [FLAGS.ext_data_source],
                                        'external_total':[external_total_patchs],
                                        'external_tissue': [external_tissue_count],
                                        'external_bg': [external_bg_count]})

        updated_external_datasets_summary = pd.concat([existing_external_datasets_summary, new_row_external_datasets], ignore_index=True)
        updated_external_datasets_summary.to_csv(all_external_datasets_summary_file, index=False)

    if not is_duplicate_external_validation.all():
        # Append new results to the All_Datasets_summary CSV
        new_row_external_validation = pd.DataFrame({'Model': [FLAGS.data_source + '_' + FLAGS.Level_patch],
                                        'External_Data_Source': [FLAGS.ext_data_source],
                                        'external_loss': [external_loss],  # Use iloc[0] to get the value
                                        'external_accuracy': [external_accuracy],  # Use iloc[0] to get the value
                                        'external_AUC': [auc_external],
                                        'external_F1score': [f1_score_external],
                                        'recall_score_external': [recall_score_external]})

        updated_external_validation_results = pd.concat([existing_external_validation_results, new_row_external_validation], ignore_index=True)
        updated_external_validation_results.to_csv(All_external_validation_results_file, index=False)
