import os
import cv2
import numpy as np
from joblib import Parallel, delayed
import torchvision.transforms as transforms
import torch

# Define the transformation to be applied to each image


def calculate_channel_stats(path_info_dir):
    transform = transforms.ToTensor()

    def process_image(patch_path):
        image = cv2.imread(patch_path)
        if image is None:
            # Handle case when the image cannot be read
            return None

        # Apply the transformation to convert the image to a tensor
        tensor_image = transform(image)

        # Calculate the mean and std of the tensor image
        tensor_mean = torch.mean(tensor_image, dim=(1, 2))
        tensor_std = torch.std(tensor_image, dim=(1, 2))

        return tensor_mean, tensor_std

    # Read image paths from the text file
    with open(path_info_dir, 'r') as file:
        # Skip the header line
        next(file)

        # Initialize lists to store tensor means and stds
        tensor_means_list = []
        tensor_stds_list = []

        # Use joblib's Parallel function to distribute the workload across multiple CPU cores
        results = Parallel(n_jobs=-1)(
            delayed(process_image)(line.strip().split(",")[4]) for line in file
        )

        for tensor_mean, tensor_std in results:
            tensor_means_list.append(tensor_mean)
            tensor_stds_list.append(tensor_std)

    # Convert the lists of tensor means and stds to NumPy arrays
    tensor_means = torch.stack(tensor_means_list).numpy()
    tensor_stds = torch.stack(tensor_stds_list).numpy()

    # Calculate the mean and standard deviation for each channel separately
    means = np.mean(tensor_means, axis=0)
    stds = np.mean(tensor_stds, axis=0)

    return list(means), list(stds)  # Convert the NumPy arrays to Python lists

