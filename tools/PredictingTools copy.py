import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imageio
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
sys.path.append(os.getcwd())
from tools.CroppingTools import crop_for_predict



class Dataset(Dataset):
	def __init__(self, crop_list, coordinates_list, channel_means, channel_stds):
		self.patch_files = crop_list
		self.patch_coordinates_list = coordinates_list
		self.channel_means = channel_means
		self.channel_stds = channel_stds

		self.augmentation = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=self.channel_means, std=self.channel_stds),
			#transforms.CenterCrop(32),
			])
	def __len__(self):
		return len(self.patch_files)
	
	def __getitem__(self, idx):
		img_arr = self.patch_files[idx]
		(x_coord, ycoord) = self.patch_coordinates_list[idx]
		augmented_image = self.augmentation(img_arr)
		
		return augmented_image, x_coord, ycoord



def calculate_channel_stats(img_list):
	transform = transforms.ToTensor()
	def process_image(img_arr):
		if img_arr is None:
			# Handle case when the image cannot be read
			return None
		
		tensor_image = transform(img_arr)
		
		# Calculate the mean and std of the tensor image
		tensor_mean = torch.mean(tensor_image, dim=(1, 2))
		tensor_std = torch.std(tensor_image, dim=(1, 2))
		
		return tensor_mean, tensor_std

	tensor_means_list = []
	tensor_stds_list = []
	
	results = Parallel(n_jobs=-1)(
		delayed(process_image)(img_arr) for img_arr in img_list
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

	return means, stds



def test_model(model, device, data_loader):
	with torch.no_grad():
		x_coords, y_coords, predictions, probabilities = list(), list(), list(), list()
		
		for inputs, x_coord, y_coord in data_loader:
			inputs = inputs.to(device)
			predicted = model(inputs)
			_,prediction = torch.max(predicted, 1)
			probability = torch.softmax(predicted, dim=1)
			for i in range(len(prediction)):
				x_coords.append(x_coord[i].item())
				y_coords.append(y_coord[i].item())
				predictions.append(prediction[i].item())
				probabilities.append(probability[i][1].item())

	del inputs, predicted, prediction, probability
	return x_coords, y_coords, predictions, probabilities


# def chooseModel(model_folder_pth):
# 	"""
# 	returns model_path, model_cropsize, means, stds
# 	"""
# 	model_folder = os.listdir(model_folder_pth)
# 	for file in model_folder:
# 		if file.endswith('.pth'):
# 			pth_file = file
# 			break
# 	model_path = os.path.join(model_folder_pth, pth_file)
# 	model_level = int(model_folder_pth.split('_')[-2].strip('L'))
# 	model_cropsize = int(model_folder_pth.split('_')[-1])
# 	means = np.load(os.path.join(model_folder_pth, 'means.npy'))
# 	stds = np.load(os.path.join(model_folder_pth, 'stds.npy'))
	
# 	return model_path, model_level, model_cropsize, means, stds


def chooseModel(data_source, Level_patch, model_type):
    """
    returns model_path, model_cropsize, means, stds
    """
    model_path = f'./results/{data_source}/{Level_patch}/trained_models/{model_type}.pth'
    dataset_summary_txt = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/Train_Val_Dataset_summary.txt'
    with open(dataset_summary_txt, "r") as file:
        header = file.readline()
        line = file.readline().replace(" ", "")
        line = line.replace('"', "'").replace("[", "").replace("]", "")
        line = line.replace("'", "")
        line = line.split(",")
        means = line[9:12] # this means 9th, 10th and 11th elements
        means = np.array([float(mean) for mean in means])
        stds = line[12:15] # this means 12th, 13th and 14th elements
        stds = np.array([float(std) for std in stds])
    
    
    return model_path, means, stds
	
def LoadData(img_arr, cropsize, stride, means, stds):
	
	patch_list, coordinates_list = crop_for_predict(img_arr, cropsize, stride)
	
	
	dataset = Dataset(patch_list, coordinates_list, means, stds)
	dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
	
	del patch_list, coordinates_list, means, stds
	return dataloader



def TestToDataframe(model, device, dataloader):
	xcord, ycord, predictions, probabilities = test_model(model, device, dataloader)
	predictionlar = pd.DataFrame({'x_coord': xcord, 'y_coord': ycord, 'predictions': predictions, 'probabilities': probabilities})
	
	del xcord, ycord, predictions, probabilities
	return predictionlar



def SaveMask(predictionlar, img_height, img_width, cropsize, mask_path, class_weights = (0.75, 0.25), voting=None):
	"""
	
	mask_path: path to save the mask with the name of the slide
	class_weights: (background, tissue), weights for the voting
	
	voting: "hard", "soft" or None
	
	"""
	
	
	mask_arr_votes = np.zeros((img_height + 2*cropsize , img_width + 2*cropsize), dtype=np.float32)
	
	
	if voting == None:
		for i in range(len(predictionlar)):
			mask_patch = mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2].copy()
			mask_patch[mask_patch == 0] = (1 - predictionlar['predictions'][i]) * 255
			mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2] = mask_patch #np.where(mask_arr == 0, (1 - predictionlar['predictions'][i]) * 255, mask_arr)
		
	
	elif voting == "hard":
		for i in range(len(predictionlar)):
			mask_patch_voting = mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2].copy()
			mask_patch_voting = class_weights[0] if predictionlar['predictions'][i] == 0 else - class_weights[1]
			mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2] += mask_patch_voting #np.where(mask_arr == 0, (1 - predictionlar['predictions'][i]) * 255, mask_arr)
	
	elif voting == "soft":
		for i in range(len(predictionlar)):
			mask_patch_voting = mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2].copy()
			mask_patch_voting = class_weights[0]*predictionlar['probabilities'][i] if predictionlar['predictions'][i] == 0 else - class_weights[1]*predictionlar['probabilities'][i]
			mask_arr_votes[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2] += mask_patch_voting
	else:
		raise ValueError("voting must be 'hard', 'soft' or None")
	
	mask_arr_votes = mask_arr_votes[cropsize:img_height + cropsize, cropsize:img_width + cropsize]
	mask_arr = np.where(mask_arr_votes > 0, 255, 0)
	mask_arr = mask_arr.astype(np.uint8)
	mask_img = Image.fromarray(mask_arr, mode="L")
	mask_img.save(mask_path)
	
	del mask_arr, mask_img
	return mask_path





def SaveMask_old(predictionlar, img_height, img_width, cropsize, mask_path, class_weights = (0.75, 0.25), voting=None):
	"""
	mask_path: path to save the mask with the name of the slide
	
	"""
	
	
	mask_arr = np.zeros((img_height + cropsize , img_width + cropsize), dtype=np.uint8)
	
	
	for i in range(len(predictionlar)):
		mask_patch = mask_arr[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2].copy()
		mask_patch[mask_patch == 0] = (1 - predictionlar['predictions'][i]) * 255
		mask_arr[predictionlar['x_coord'][i] + cropsize:predictionlar['x_coord'][i] + cropsize*2 , predictionlar['y_coord'][i] + cropsize:predictionlar['y_coord'][i] + cropsize*2] = mask_patch #np.where(mask_arr == 0, (1 - predictionlar['predictions'][i]) * 255, mask_arr)
	
	mask_arr = mask_arr[cropsize:img_height + cropsize, cropsize:img_width + cropsize]
	mask_img = Image.fromarray(mask_arr, mode="L")
	mask_img.save(mask_path)
	
	del mask_arr, mask_img
	return mask_path



def get_eroded_mask(mask, kernel_size=2, iterations=1, inverted=True):
	if inverted:
		mask = cv2.bitwise_not(mask)
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
	if inverted:
		eroded_mask = cv2.bitwise_not(eroded_mask)
	return eroded_mask



def heatmap(predictionlar, img_height, img_width, cropsize, save_path):

    cmap = plt.get_cmap('jet')
    cNorm  = mcolors.Normalize(vmin=0, vmax=1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    
    heatmap = np.zeros((img_height, img_width), dtype=np.float32)
    heatmap_color_arr = np.zeros((img_height, img_width,4),dtype=np.float32)

    for i in range(len(predictionlar)):
                x_start, y_start = predictionlar['x_coord'][i], predictionlar['y_coord'][i]
                prob_map = predictionlar['probabilities'][i]  # Probability map for this prediction
                heatmap[x_start:x_start + cropsize, y_start:y_start + cropsize] = prob_map
                temp_color1 = scalarMap.to_rgba(prob_map)
                heatmap_color_arr[x_start:x_start + cropsize, y_start:y_start + cropsize] = temp_color1
                
    # plt.imshow(heatmap_color_arr)
    # plt.show()
    
    figure_size = ((3/img_height)*img_width*1.05 + 0.1 + 0.6,3)
    right_val = 1 - 0.6/figure_size[0]
    fig1, ax1 = plt.subplots(figsize=figure_size)
    im1 = ax1.imshow(heatmap_color_arr)
    ax1.set_xticks([])
    ax1.set_yticks([])
    divider1 = make_axes_locatable(ax1)
    
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = ax1.figure.colorbar(scalarMap, cax=cax1)
    cbar1.ax.set_ylabel("probabilities", rotation=-90, va="bottom")
    
    fig1.tight_layout()
    fig1.subplots_adjust(left=0.01, bottom=0.02,right=right_val, top=0.98, wspace=0.20, hspace=0.20)
    fig1.savefig(save_path, dpi=200)