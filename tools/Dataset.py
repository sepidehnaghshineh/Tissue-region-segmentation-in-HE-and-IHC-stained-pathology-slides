import os
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data as data
from tools.dataProperties import calculate_channel_stats
import torch
# Define a custom dataset for the cropped patches

class Dataset(Dataset):
    def __init__(self, dataset_dir, model_mode, channel_means, channel_stds):
        self.patch_files = []
        self.patch_label = []
        self.model_mode = model_mode

        with open(dataset_dir, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespaces or newlines
                parts = line.split(",")  # Split the line by comma
                patch_path = parts[4]  # the 5th  element contains the patch path
                # print(patch_path)
                
                # Extract the filename from the patch path
                patch_label = parts[3]
                patch_label = 1 if 'tissue' in patch_label else 0

                self.patch_files.append(patch_path)
                self.patch_label.append(patch_label)
        
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.aug =  transforms.Compose([
                transforms.ToTensor()])
        # Define data augmentation transformations
        if model_mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.channel_means, std=self.channel_stds),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                
            ])
            
        else:  # For validation and test and external_test modes
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.channel_means, std=self.channel_stds),
            ])
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        # assert os.path.exists(patch_path)
        image = cv2.imread(patch_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        augmented_image = self.augmentation(image)
        patch_label = self.patch_label[idx]
    

        return patch_path, augmented_image, patch_label


class Reg_Dataset(Dataset):
    def __init__(self, dataset_dir, model_mode, channel_means, channel_stds):
        self.patch_files = []
        self.tissue_ratios = []
        self.model_mode = model_mode

        with open(dataset_dir, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespaces or newlines
                parts = line.split(",")  # Split the line by comma
                patch_path = parts[4]  # the 3rd  element from last contains the patch path
                # print(patch_path)
                
                # Extract the filename from the patch path
                tissue_ratio = parts[5]
                tissue_ratio = float(tissue_ratio)

                self.patch_files.append(patch_path)
                self.tissue_ratios.append(tissue_ratio)
        
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.aug =  transforms.Compose([
                transforms.ToTensor()])
        # Define data augmentation transformations
        if model_mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.channel_means, std=self.channel_stds),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:  # For validation and test modes
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.channel_means, std=self.channel_stds),
            ])
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        # assert os.path.exists(patch_path)
        image = cv2.imread(patch_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        augmented_image = self.augmentation(image)
        # tissue_ratio = self.tissue_ratios[idx]
        tissue_ratio = torch.as_tensor(self.tissue_ratios[idx], dtype=torch.float64)#for the regression model , other wise use : tissue_ratio = self.tissue_ratios[idx]
    

        return patch_path, augmented_image, tissue_ratio

class UNET_dataset(data.Dataset):
    def __init__(self, dataset_dir, model_mode):
        self.patch_dirs = []
        self.label_dirs = []
        self.model_mode = model_mode

        df = pd.read_csv(dataset_dir)
        for row in range(len(df)):
            patch_dir = df.iloc[row]['patch_path']
            label_dir = df.iloc[row]['mask_patch_path']
            self.patch_dirs.append(patch_dir)
            self.label_dirs.append(label_dir)
        
        if model_mode == 'train':
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                
            ])
        else:  # For validation and test modes
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),
            ])
        
    def __len__(self):
        return len(self.patch_dirs)
    
    def __getitem__(self, idx):
        patch_dir = self.patch_dirs[idx]
        label_dir = self.label_dirs[idx]
        # assert os.path.exists(patch_dir)
        # assert os.path.exists(label_dir)
        patch = cv2.imread(patch_dir)
        label_img = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        augmented_patch = self.augmentation(patch)
        label_img = transforms.ToTensor()(label_img)
        return patch_dir, augmented_patch, label_img