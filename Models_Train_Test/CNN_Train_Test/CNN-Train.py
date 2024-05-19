import os
import sys
sys.path.append("./")
# sys.path.append(os.getcwd())
import torch
import argparse
import pandas as pd
import numpy as np
import csv
import torch.nn as nn
from tqdm import tqdm
# from tools.Dataset_w_hue import Dataset
from tools.Dataset import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tools.dataProperties import calculate_channel_stats
from tools.prepare_dataset import read_patch_paths
from tools.prepare_dataset import dataset_info
from torch.utils.data import DataLoader
from tools.LeNet5_different_inputSizes import select_model
import time

print(torch.__version__)
parser = argparse.ArgumentParser()

# Arguments to be provided before running the code.
parser.add_argument('--train_dataset_dir', default='./Dataset_Prep/data_preparation_results/Cropped_slides/source_name/train/allpatches__level_4_size_128.txt', help='train dataset info folder', dest='train_dataset_dir')
parser.add_argument('--val_dataset_dir', default='./Dataset_Prep/data_preparation_results/Cropped_slides/source_name/val/allpatches__level_4_size_128.txt', help='vall dataset info folder', dest='val_dataset_dir')
parser.add_argument('--data_source', default='source_name', help='the source that you have got the WSIs', dest='data_source')
parser.add_argument('--Level_patch', default='L4_128', help='different levels from 4 to 6 and different patch sizes from 16 to 128', dest='Level_patch')
parser.add_argument('--batch_size', default='1024', type=int, help='batch size', dest='batch_size')
parser.add_argument('--input_size', default='128', type=int, help='different input size for selecting the proper LeNet5 model which could be one of  16,32,64,128 input_sizes', dest='input_size')
parser.add_argument('--learning_rate', default='0.0001', type=float, help='number of patches each patient has', dest='learning_rate')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of steps of execution', dest='num_epochs')
parser.add_argument('--model_mode', default='train', help='model_mode', dest='model_mode')
parser.add_argument('--model_type', default='LeNet5', help='the used CNN model', dest='model_type')
parser.add_argument('--device_num',default='1', type= int, help= 'cuda is either 0 or 1', dest= 'device_num' )

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

if not os.path.exists('./results'):
    os.mkdir('./results')

if not os.path.exists('./results/{}'.format(FLAGS.data_source)):
    os.mkdir('./results/{}'.format(FLAGS.data_source))

if not os.path.exists('./results/{}/{}'.format(FLAGS.data_source,FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}'.format(FLAGS.data_source,FLAGS.Level_patch))

if not os.path.exists('./results/{}/{}/train_test_summary'.format(FLAGS.data_source, FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}/train_test_summary'.format(FLAGS.data_source,FLAGS.Level_patch))


if not os.path.exists('./results/{}/{}/trained_models'.format(FLAGS.data_source,FLAGS.Level_patch)):
    os.mkdir('./results/{}/{}/trained_models'.format(FLAGS.data_source,FLAGS.Level_patch))

if not os.path.exists('./results/{}/{}/train_test_summary/{}'.format(FLAGS.data_source,FLAGS.Level_patch, FLAGS.model_type)):
    os.mkdir('./results/{}/{}/train_test_summary/{}'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type))


# Define the path to the cropped patches directory
train_dataset_dir = FLAGS.train_dataset_dir
val_dataset_dir = FLAGS.val_dataset_dir
# Calculate channel mean and std for both datasets
train_mean, train_std = calculate_channel_stats(train_dataset_dir)
model = select_model(FLAGS.input_size)
# print(model)

#################################################################################

def save_summary_info(file_path, best_epoch, train_patch_count, class_names, train_labels_count, val_patch_count, val_labels_count, train_mean, train_std):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["data_source", "Best_Epoch", "Train_total_patchs", "Train_Label_names", "Train_Label_counts",
                         "Val_total_patchs", "Val_Label_counts", "Train_Channel_Means", "Train_Channel_Stds"])
        
        # Write data
        writer.writerow([FLAGS.data_source + '_' + FLAGS.Level_patch , best_epoch, train_patch_count, class_names, train_labels_count,
                         val_patch_count, val_labels_count, train_mean, train_std])

#################################################################################################################################

def train_model(model, device, optimizer, criterion, dataloader):

    train_loss, train_accuracy, total = 0, 0, 0

    train_pbar = tqdm(total=len(dataloader))

    model.train()
    # Instantiate the CNN model
    for pathes, patchs, labels in dataloader:
              train_pbar.update(1) # progress par.
              patchs, labels = patchs.to(device), labels.to(device) # get the input and real species as outputs;
              optimizer.zero_grad() # zero the parameter gradients 
              outputs = model(patchs) # predict output from the model
              loss = criterion(outputs, labels) # calculate loss for the predicted output
              loss.backward() # backpropagate the loss 
              optimizer.step() # adjust parameters based on the calculated gradients 
              train_loss += loss.item()  # track the loss value 
              #print(loss.item())
              total += labels.size(0) # track number of predictions
              _, predicted = torch.max(outputs, 1) # The label with the highest value will be our prediction 
              train_accuracy += (predicted == labels).sum().item() # number of matched predictions
    train_pbar.close()
    # Calculate loss as the sum of loss in each batch divided by the total number of predictions done.  
    train_loss = train_loss/len(dataloader)
    # Calculate accuracy as the number of correct predictions in the batch divided by the total number of predictions done.  
    train_accuracy = (100 * train_accuracy /total)

    return model, train_loss, train_accuracy
     
 ######################################################################################################################            

def validate_model(model, device, dataloader, class_names):  
    model.eval()

    patch_paths, y_true, y_pred ,y_prob_raw = list(), list(), list(), list() 

    vall_loss, vall_accuracy, total = 0, 0, 0

    validate_pbar = tqdm(total=len(dataloader)) 
    criterion = nn.CrossEntropyLoss() # define the loss function


    with torch.no_grad():
        for paths,patchs, labels in dataloader: 
            validate_pbar.update(1) 

            patchs, labels = patchs.to(device), labels.to(device) # get the input and real species as outputs;
            outputs = model(patchs) # predict output from the model

            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1) # dim=1 is the dimension of the classes

            loss = criterion(outputs, labels) # calculate loss for the predicted output

            _, predicted = torch.max(outputs, 1) # The label with the highest value will be our prediction

            total += labels.size(0) # track number of predictions

            vall_loss += loss.item() # track the loss value

            vall_accuracy += (predicted == labels).sum().item() # number of matched predictions

            for i in range(len(predicted)):
                patch_paths .append(paths[i])
                y_true.append(class_names[labels[i].item()])  # Use class names instead of integer labels
                y_prob_raw.append(probabilities[i][1].item())  # Use the probability of the positive class (tissue)
                y_pred.append(class_names[predicted[i].item()]) # Use class names instead of integer labels
                
        validate_pbar.close() 

        vall_loss = vall_loss/len(dataloader) # Calculate loss as the sum of loss in each batch divided by the total number of predictions done.    
        vall_accuracy = (100*vall_accuracy/ total) # Calculate accuracy as the number of correct predictions in the batch divided by the total number of predictions done.

    validation_results_dic = { 'patch_path': patch_paths,'y_true': y_true, 'y_pred': y_pred, 'y_prob_raw': y_prob_raw} # Create a dictionary to store the results
    validation_results = pd.DataFrame(validation_results_dic, columns=['patch_path','y_true', 'y_pred', 'y_prob_raw']) # Create a dataframe to store the results

    return vall_loss, vall_accuracy, validation_results  

###########################################################################################################################
       
def train_val_model(model, device, optimizer, criterion, train_loader, val_loader, class_names, num_epochs):
    train_loss_list = []
    train_accuracy_list = []
    vall_loss_list = []
    vall_accuracy_list = []
    


    best_epoch = 0
    best_accuracy = 0 
    num_epochs = FLAGS.num_epochs


    
    for epoch in range(num_epochs):
        start =  time.time()
        
        print('############## EPOCH - {} ##############'.format(epoch+1))
        
        # train model
        print('******** training ******** \n')      
        
        model, train_loss, train_accuracy = train_model(model, device, optimizer, criterion, train_loader)
        print('tarin_loss= {:.3f}, train_accuracy= {:.3f}% \n'.format(train_loss, train_accuracy))

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        
        # validate model
        print('******** validating ******** \n')
        vall_loss, vall_accuracy, validation_results = validate_model(model, device, val_loader, class_names)
        print('vall_loss= {:.2f}, vall_accuracy= {:.2f}% \n'.format(vall_loss, vall_accuracy))

        
        vall_loss_list.append(vall_loss)
        vall_accuracy_list.append(vall_accuracy)



        if epoch % 5== 0:
            train_data = pd.DataFrame({'loss': train_loss_list, 'accuracy': train_accuracy_list}, columns= ['loss', 'accuracy'])
            val_data = pd.DataFrame({'loss': vall_loss_list, 'accuracy': vall_accuracy_list}, columns= ['loss', 'accuracy'])
            train_data.to_csv('./results/{}/{}/train_test_summary/{}/train_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type), header= True)
            val_data.to_csv('./results/{}/{}/train_test_summary/{}/val_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type), header= True)
        
        
        if vall_accuracy > best_accuracy: # If the current accuracy is better than the best accuracy, save the model
            best_accuracy = vall_accuracy # Update the best accuracy
            best_epoch = epoch + 1  # Epochs are 0-indexed, so add 1 to get the actual epoch number
            print('saving model with best accuracy of: {:.3f}% \n'.format(best_accuracy))
            torch.save(model.state_dict(), './results/{}/{}/trained_models/{}.pth'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)) # Save the model with the best accuracy
        end = time.time()
        print('time :' +  str(end - start ) +'\n')
    # Specify the file path where you want to save the summary
    summary_file_path = './results/{}/{}/train_test_summary/{}/Train_Val_Dataset_summary.txt'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)

    # Save the summary information to the specified text file
    save_summary_info(summary_file_path, best_epoch, train_patch_count, class_names, train_labels_count, val_patch_count, val_labels_count, train_mean, train_std)


    
    # saving data into csv        
    train_data = pd.DataFrame({'loss': train_loss_list, 'accuracy': train_accuracy_list}, columns= ['loss', 'accuracy'])
    val_data = pd.DataFrame({'loss': vall_loss_list, 'accuracy': vall_accuracy_list}, columns= ['loss', 'accuracy'])
    
    train_data.to_csv('./results/{}/{}/train_test_summary/{}/train_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type), header= True)
    val_data.to_csv('./results/{}/{}/train_test_summary/{}/val_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type), header= True)
    
    validation_results.to_csv('./results/{}/{}/train_test_summary/{}/validation_results.csv'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type), header= True)


random_seed = 42 # Set the random seed for reproducibility

np.random.seed(random_seed) 

print('Preparing datasets: ...')


train_patchs, train_labels, train_labels_count, class_names , tissue_ratio, bg_ratio , train_base_coords, train_mask_coords, train_mask_lvl= dataset_info(FLAGS.train_dataset_dir) 
val_patchs, val_labels, val_labels_count, class_names, val_tissue_ratio, val_bg_ratio , val_base_coords, val_mask_coords, val_mask_lvl= dataset_info(FLAGS.val_dataset_dir)

# Use the function to save the summary information
train_patch_count = len(train_patchs)
val_patch_count = len(val_patchs)


train_dataset = Dataset(train_dataset_dir, FLAGS.model_mode, train_mean, train_std)
train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=30)   # num_workers=40 is the number of CPU cores to use for data loading

val_dataset = Dataset(val_dataset_dir,"val",train_mean, train_std )
val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=30)

device = torch.device(f"cuda:{FLAGS.device_num}" if torch.cuda.is_available() else "cpu")

print('cuda{} is available: {}\n'.format(FLAGS.device_num, torch.cuda.is_available()))
print(f'Moving model to device {FLAGS.device_num}')
model.to(device)
        
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9, weight_decay=0.0001) # Use stochastic gradient descent (SGD) as the optimizer with a learning rate of 0.0001, momentum of 0.9, and weight decay of 0.0001

# start training and evaluation
train_val_model(model, device, optimizer, criterion, train_dataloader, val_dataloader, class_names ,num_epochs=FLAGS.num_epochs )
vall_loss, vall_accuracy, validation_results = validate_model(model, device, val_dataloader, class_names)
# plot charts:

train_data = pd.read_csv('./results/{}/{}/train_test_summary/{}/train_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch,FLAGS.model_type)) 
val_data = pd.read_csv('./results/{}/{}/train_test_summary/{}/val_data.csv'.format(FLAGS.data_source, FLAGS.Level_patch, FLAGS.model_type))
