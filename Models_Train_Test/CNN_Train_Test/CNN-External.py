import os
import torch
import sys
sys.path.append("./")
import csv
import torch.nn as nn
import ast  # For safely evaluating the string representation of lists
import numpy as np
import argparse
import pandas as pd
import torch.utils.data
from tools.Dataset import Dataset
from tqdm import tqdm
from tools.Dataset import Dataset
from torch.utils.data import DataLoader
from tools.LeNet5_different_inputSizes import select_model
import matplotlib.image as mpimg
from tools.prepare_dataset import dataset_info , read_patch_paths



# Define class label mapping
class_label_mapping = {'background': 0, 'tissue': 1}
parser = argparse.ArgumentParser(
    description='Test LeNet5 ')


# Arguments to be provided before running the code.
parser.add_argument('--dataset_dir', default='./Dataset_Prep/data_preparation_results/Cropped_slides/source_name/test/allpatches__level_4_size_128.txt',help='testing dataset path', dest='dataset_dir')   
parser.add_argument('--data_source', default='source_name', help='the source of WSIs that you have trained the model with', dest='data_source')
parser.add_argument('--ext_data_source', default='external_source_name', help='the source of WSIs that you want to evaluate and test the trained model with', dest='ext_data_source')
parser.add_argument('--Level_patch', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch')
parser.add_argument('--batch_size', default='1024', type=int,help='batch size', dest='batch_size')
parser.add_argument('--model_type', default='LeNet5',help='', dest='model_type')
parser.add_argument('--model_mode', default='external_test', help='model_mode', dest='model_mode')
parser.add_argument('--input_size', default='128', type=int, help='different input size for selecting the proper LeNet5 model which could be one of  16,32,64,128 input_sizes', dest='input_size')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

if not os.path.exists('./results/external'):
    os.mkdir('./results/external')

if not os.path.exists('./results/external/{}'.format(FLAGS.data_source)):
    os.mkdir('./results/external/{}'.format(FLAGS.data_source))

####################################################################
def save_summary_info(file_path,external_patch_count, patch_label, external_labels_count, train_mean, train_std):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["Model","external_data_source", "external_total_patchs", "external_Label_names", "external_Label_counts",
                         "external_Channel_Means", "external_Channel_Stds"])
        
        # Write data
        writer.writerow([FLAGS.data_source + '_' + FLAGS.Level_patch,FLAGS.ext_data_source + '_' + FLAGS.Level_patch ,external_patch_count, patch_label,external_labels_count,
                          train_mean, train_std])

#################################################################################
def read_summary_info(file_path):
     # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract the values from the 'Train Channel Means' and 'Train Channel Stds' columns
    train_channel_means = ast.literal_eval(df['Train_Channel_Means'][0])
    train_channel_stds = ast.literal_eval(df['Train_Channel_Stds'][0])

    return train_channel_means, train_channel_stds

####################################################################################
Train_Val_Dataset_summary_file_path = './results/{}/{}/train_test_summary/{}/Train_Val_Dataset_summary.txt'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)
train_mean, train_std = read_summary_info(Train_Val_Dataset_summary_file_path)
#trained_models_path = '/home/snaghshineh/Documents/TissueSegmentation/results/HER2_IHC/L5_64/RUN_1/trained_models/LeNet5_just_Student.pth'
trained_models_path = './results/{}/{}/trained_models/'.format(FLAGS.data_source, FLAGS.Level_patch)
criterion = nn.CrossEntropyLoss()
model = select_model(FLAGS.input_size)
# print(model)

def external_test_model(model, device, data_loader, patch_label):  
    model.eval()
    total, running_accuracy, test_loss = 0, 0, 0
    patch_path, y_true, y_pred ,y_prob_raw = list(), list(), list(), list()
    
    test_pbar = tqdm(total=len(data_loader))

    with torch.no_grad():
        for paths, patchs, labels in data_loader:
            test_pbar.update(1)
       
            inputs, labels = patchs.to(device), labels.to(device)
            # calculate labels by running patchs through the network
            predicted_labels = model(inputs)
                        # Apply softmax to get probabilities
            probabilities = torch.softmax(predicted_labels, dim=1)

            loss = criterion(predicted_labels, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(predicted_labels, 1)
            
            total += labels.size(0)

            test_loss += loss.item()

            running_accuracy += (predicted == labels).sum().item() 

            for i in range(len(labels)):
                patch_path.append(paths[i])
                y_true.append(patch_label[labels[i]])
                y_prob_raw.append(probabilities[i][1].item())  # Use the probability of the positive class (tissue)
                y_pred.append(patch_label[predicted[i]])



        test_pbar.close()

        test_loss = test_loss/len(data_loader)
        test_acc = (100 * running_accuracy / total)
    test_data = pd.DataFrame({'loss': [test_loss], 'accuracy': [test_acc]}, columns= ['loss', 'accuracy'])

    test_results_dic = { 'patch_path': patch_path,'y_true': y_true, 'y_pred': y_pred, 'y_prob_raw': y_prob_raw}
    test_results = pd.DataFrame(test_results_dic, columns=['patch_path','y_true', 'y_pred', 'y_prob_raw'])
    print('Loss of the model based on the test set of ', len(data_loader) ,' inputs is: %d' % test_loss)    
    print('Accuracy of the model based on the test set of ', len(data_loader) ,' inputs is: %d %%' % test_acc)    
    return  test_data, test_results

# test_vall_dasets_dirs = '' 

random_seed = 42

np.random.seed(random_seed)

print('Preparing testing datasets: ...')

 
test_dataset_dirs = FLAGS.dataset_dir

external_patch, external_labels, external_labels_count, patch_label ,external_tissue_ratio, external_bg_ratio,test_base_coords, external_mask_coords, external_mask_lvl= dataset_info(test_dataset_dirs)


# Use the function to save the summary information
external_patch_count = len(external_patch)

# Specify the file path where you want to save the summary
summary_file_path = './results/external/{}/external_Dataset_summary.txt'.format(FLAGS.data_source)

# Save the summary information to the specified text file
save_summary_info(summary_file_path, external_patch_count, patch_label, external_labels_count, train_mean, train_std)

test_dataset = Dataset(test_dataset_dirs, FLAGS.model_mode, train_mean, train_std)
test_dataloader = DataLoader(test_dataset, batch_size= FLAGS.batch_size, num_workers= 0 , shuffle= "False")


trained_model_path = trained_models_path + '{}.pth'.format(FLAGS.model_type)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(trained_model_path, map_location=device))

print('cuda is available: {} \n'.format(torch.cuda.is_available()))
print('moving model to available device')
model.to(device)


# start testing
external_data, external_results= external_test_model(model, device, test_dataloader, patch_label)

external_results.to_csv('./results/external/{}/{}_external_val_results.csv'.format(FLAGS.data_source,FLAGS.ext_data_source), header= True)
external_data.to_csv('./results/external/{}/{}_external_val_data.csv'.format(FLAGS.data_source, FLAGS.ext_data_source), header= True)
external_results = pd.read_csv('./results/external/{}/{}_external_val_results.csv'.format(FLAGS.data_source, FLAGS.ext_data_source))
