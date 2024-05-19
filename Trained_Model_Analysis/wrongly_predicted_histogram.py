import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser()

# Arguments to be provided before running the code.
parser.add_argument('--data_source', default='source_name', help='TCGA  or Camelyon17 or HER2 or HER2OHE or BAU', dest='data_source')
parser.add_argument('--Level_patch', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch')
parser.add_argument('--level', default='4', type= int,help='different levels from 4 to 6', dest='level')
parser.add_argument('--patch', default='128', type=int, help='different patch sizes from 16 to 128', dest='patch')
parser.add_argument('--model_mode', default='train', help='model_mode', dest='model_mode')
parser.add_argument('--model_type', default='LeNet5', help='CNN name', dest='model_type')
parser.add_argument('--val_test',default='test', help='the directory is either named validation or test or validating or testing or Test or Testing or Validation or Validationg', dest='val_test')
FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)



def find_patches_in_wrong_results(allpatches_text_file, wrong_results_file, output_file_tissue, output_file_background,output_file_high_tissue):
    # Read the crops info CSV file
    allpatches_df = pd.read_csv(allpatches_text_file)

    # Read the test_Wrong_results CSV file
    wrong_results_df = pd.read_csv(wrong_results_file)

    # Filter rows in allpatches_df that have 'patch_path' values present in wrong_results_df
    # result_df = allpatches_df[allpatches_df['patch_path'].isin(wrong_results_df['patch_path'])]
    result_df = allpatches_df[allpatches_df.iloc[:, 4].isin(wrong_results_df['patch_path'])]


    high_tissue_patches_df = result_df[(result_df.iloc[:, 3] == 'tissue') & (result_df.iloc[:, 5] > 0.8)]
    high_tissue_patches_df.to_csv(output_file_high_tissue, index=False)
    # Split the result into two DataFrames based on 'y_true' column
    tissue_patches_df = result_df[result_df.iloc[:, 3] == 'tissue']
    background_patches_df = result_df[result_df.iloc[:, 3] == 'background']

    # Save the two DataFrames to separate CSV files
    tissue_patches_df.to_csv(output_file_tissue, index=False)
    background_patches_df.to_csv(output_file_background, index=False)
    # # Save the result to a new CSV file
    # result_df.to_csv(output_file, index=False)

def create_histograms_and_save_subplots(directory1, directory2, save_directory):
    # Load the datasets from both directories
    df1 = pd.read_csv(directory1)
    df2 = pd.read_csv(directory2)

    # Extract the 'tissue_ratio' columns
    tissue_ratios1 = df1.iloc[:, 5] * 100  # Convert to percentage
    tissue_ratios2 = df2.iloc[:, 5] * 100  # Convert to percentage

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create the histogram for tissue_ratio from directory1
    axes[0].hist(tissue_ratios1, bins=10, range=(0, 100), edgecolor='black')
    axes[0].set_xlabel('Tissue Ratio (%)')
    axes[0].set_ylabel('Number of Patches')
    axes[0].set_title('Tissue Ratio Distribution - wrongly_predicted_tissues')

    # Create the histogram for tissue_ratio from directory2
    axes[1].hist(tissue_ratios2, bins=10, range=(0, 100), edgecolor='black')
    axes[1].set_xlabel('Tissue Ratio (%)')
    axes[1].set_ylabel('Number of Patches')
    axes[1].set_title('Tissue Ratio Distribution - wrongly_predicted_backgrounds')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Define the path to save the subplot
    save_path = save_directory

    # Save the subplot to the desired directory
    plt.savefig(save_path)

    # Display the subplot (optional)
    plt.show()

def create_histograms_bg_ratio(directory1, directory2, save_directory):
    # Load the datasets from both directories
    df1 = pd.read_csv(directory1)
    df2 = pd.read_csv(directory2)

    # Extract the 'bg_ratio' columns
    background_ratios1 = df1.iloc[:, 6] * 100  # Convert to percentage
    background_ratios2 = df2.iloc[:, 6] * 100  # Convert to percentage

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create the histogram for tissue_ratio from directory1
    axes[0].hist(background_ratios1, bins=10, range=(0, 100), edgecolor='black')
    axes[0].set_xlabel('background Ratio (%)')
    axes[0].set_ylabel('Number of Patches')
    axes[0].set_title('background Ratio Distribution - wrongly_predicted_tissue_labeled_patchs')

    # Create the histogram for tissue_ratio from directory2
    axes[1].hist(background_ratios2, bins=10, range=(0, 100), edgecolor='black')
    axes[1].set_xlabel('background Ratio (%)')
    axes[1].set_ylabel('Number of Patches')
    axes[1].set_title('background Ratio Distribution - wrongly_predicted_backgrounds_labeled_patchs')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Define the path to save the subplot
    save_path = save_directory

    # Save the subplot to the desired directory
    plt.savefig(save_path)

    # Display the subplot (optional)
    plt.show()



##########################################################################################################

# Specify the file paths
# /home/snaghshineh/Documents/TissueSegmentation/results/TCGA/L4_128/RUN_1/train_test_summary/LeNet5/just_Student/test___Wrong_results.csv
wrong_results_file  = './TissueSegmentation/results/{}/{}/train_test_summary/{}/test___Wrong_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)

allpatches_text_file = './TissueSegmentation/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/{}/{}/allpatches__level_{}_size_{}.txt'.format(FLAGS.data_source,FLAGS.val_test, FLAGS.level, FLAGS.patch)
percentage_info_wrongly_predicted = './TissueSegmentation/results/{}/{}/train_test_summary/{}/{}/percentage_info_wrongly_predicted.csv'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
percentage_info_wrongly_predicted_tissues = './TissueSegmentation/results/{}/{}/train_test_summary/percentage_info_wrongly_predicted_tissues.csv'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
percentage_info_wrongly_predicted_backgrounds = './TissueSegmentation/results/{}/{}/train_test_summary/percentage_info_wrongly_predicted_backgrounds.csv'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
percentage_info_high_tissue_ratio = './TissueSegmentation/results/{}/{}/train_test_summary/wrongly_high_tissue_ratio.csv'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)

histogram_wrongly_predicted_tissues = './TissueSegmentation/results/{}/{}/train_test_summary/hisotgram_wrongly_tissues.png'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
histogram_wrongly_predicted_backgrounds = './TissueSegmentation/results/{}/{}/train_test_summary/hisotgram_wrongly_backgrounds.png'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
histogram_tissue_ratio_wrongly_predicted = './TissueSegmentation/results/{}/{}/train_test_summary/histogram_tissue_ratio_wrongly_predicted.png'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)
histogram_background_ratio_wrongly_predicted = './TissueSegmentation/results/{}/{}/train_test_summary/histogram_background_ratio_wrongly_predicted.png'.format(FLAGS.data_source, FLAGS.Level_patch,  FLAGS.model_type)


find_patches_in_wrong_results(allpatches_text_file, wrong_results_file, percentage_info_wrongly_predicted_tissues,percentage_info_wrongly_predicted_backgrounds, percentage_info_high_tissue_ratio)

create_histograms_and_save_subplots(percentage_info_wrongly_predicted_tissues, percentage_info_wrongly_predicted_backgrounds, histogram_tissue_ratio_wrongly_predicted)
create_histograms_bg_ratio(percentage_info_wrongly_predicted_tissues, percentage_info_wrongly_predicted_backgrounds, histogram_background_ratio_wrongly_predicted)
