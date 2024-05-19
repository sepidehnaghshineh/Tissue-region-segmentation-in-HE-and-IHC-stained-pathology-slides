from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score , roc_curve , auc
import itertools
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import ast
import csv
import re
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, auc, recall_score , precision_score


def plt_charts(train_loss_list, train_acc_list, vall_loss_list, vall_acc_list, num_epochs, path, data_source, Level_patch): 
        new_list = list(range(0, num_epochs + 1, 200))
        print('****** saving figures *******')
        # plotting the data and save figure
        plt.figure('train', (12,6))
        plt.subplot(1,2,1)
        plt.title('Epoch vs Loss')
        
        epoch = [i + 1 for i in range(len(train_loss_list))]
        
        plt.xlabel('epoch')
        plt.xticks(new_list)
        
        plt.plot(epoch,train_loss_list, label = 'Train Loss')
        plt.plot(epoch,vall_loss_list, label = 'Validation Loss')
        plt.legend(loc="best")
        
        plt.subplot(1,2,2)
        plt.title('Epoch vs Accuracy')

        plt.xlabel('epoch')
        plt.xticks(new_list)

        plt.plot(epoch, train_acc_list, label='Train Accuracy')
        plt.plot(epoch, vall_acc_list, label='Validation Accuracy')
        
        plt.legend(loc="best")
        plt.show
        plt.savefig('{}/{}_{}_loss_and_acc_vs_epoch.png'.format(path, data_source,Level_patch))
        plt.show()

##################################################################################################################
def plt_Reg_charts(train_loss_list, vall_loss_list, num_epochs, path, data_source,Level_patch): 
        new_list = list(range(0, num_epochs + 1, 100))
        print('****** saving figures *******')
        # plotting the data and save figure
        plt.figure('train', (12,6))
        plt.subplot(1,2,1)
        plt.title('Epoch vs Loss')
        
        epoch = [i + 1 for i in range(len(train_loss_list))]
        
        plt.xlabel('epoch')
        plt.xticks(new_list)
        
        plt.plot(epoch,train_loss_list, label = 'Train Loss')
        plt.plot(epoch,vall_loss_list, label = 'Validation Loss')
        plt.legend(loc="best")

        plt.legend(loc="best")
        plt.show
        plt.savefig('{}/{}_{}_loss_and_acc_vs_epoch.png'.format(path, data_source,Level_patch))



def plot_loss_curve(train_losses, val_losses, num_epochs, save_path):
    """
    Plot the training and validation loss curves.

    Args:
    train_losses (list): List of training losses for each epoch.
    val_losses (list): List of validation losses for each epoch.
    num_epochs (int): The total number of training epochs.
    save_path (str): Path to save the plot.

    Returns:
    None
    """
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, num_epochs + 1, 200))
    plt.savefig('{}/loss_vs_epoch.png'.format(save_path))
    

######################################################################################################################


def calculate_roc_and_plot(y_true, y_prob_raw):
    fpr, tpr, _ = roc_curve(y_true, y_prob_raw)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    return roc_auc

###############################################################################################################
def extract_auc_confidence(results):
    class_label_mapping = {'background': 0, 'tissue': 1}
    y_true_numeric = np.array([class_label_mapping[label] for label in results['y_true']])
    results['y_true_str'] = results['y_true'].apply(lambda x: 'tissue' if x == 1 else 'background')
    y_prob_raw = results['y_prob_raw'].values

    bootstrapped_scores, confidence_lower, confidence_upper = BootStrap(y_true_numeric, y_prob_raw, n_bootstraps=1000)

    return y_true_numeric, y_prob_raw, np.mean(bootstrapped_scores), confidence_lower, confidence_upper

#################################

def score_fnc(y_true, y_prob_raw, metric='auc'):
	if metric == 'auc':
		auc = roc_auc_score(y_true, y_prob_raw)
		return auc
	elif metric == 'recall':
		recall = recall_score(y_true, y_prob_raw > 0.5)
		return recall
	elif metric == 'f1':
		f1 = f1_score(y_true, y_prob_raw > 0.5)
		return f1
	elif metric == 'accuracy':
		accuracy = accuracy_score(y_true, y_prob_raw > 0.5)
		return accuracy
	else:
		raise ValueError(f'Metric not supported: {metric}')
	
def BootStrap(y_true, y_prob_raw, n_bootstraps, metric = 'auc'):
	"""
	Calculates the difference in scores for each bootstrap iteration
	returns sorted_scores, confidence_lower, confidence_upper
	"""
	
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []


	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(y_prob_raw), len(y_prob_raw))

		if len(np.unique(y_prob_raw[indices])) < 2:
			continue
		else:
			score = score_fnc(y_true[indices], y_prob_raw[indices], metric=metric)
			bootstrapped_scores.append(score)
		

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	return sorted_scores, confidence_lower, confidence_upper

def BootStrap_Compare(y_true, y_prob_raw1, y_prob_raw2, n_bootstraps, metric='auc' ):
	"""
	Calculates the difference in scores for each bootstrap iteration
	Returns three tuples of (sorted_scores, confidence_lower, confidence_upper)
	"""
	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores1 = []
	bootstrapped_scores2 = []
	
	
	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		indices = rng.randint(0, len(y_prob_raw1), len(y_prob_raw1))
		
		if len(np.unique(y_prob_raw1[indices])) < 2:
			continue
		else:
			score1 = score_fnc(y_true[indices], y_prob_raw1[indices], metric=metric)
			score2 = score_fnc(y_true[indices], y_prob_raw2[indices], metric=metric)
			
			bootstrapped_scores1.append(score1)
			bootstrapped_scores2.append(score2)
		
	bootstrapped_scores_diff = np.array(bootstrapped_scores1) - np.array(bootstrapped_scores2)
	sorted_scores1 = np.array(bootstrapped_scores1)
	sorted_scores2 = np.array(bootstrapped_scores2)
	sorted_scores_diff = np.array(bootstrapped_scores_diff)
	
	confidence_lower1 = sorted_scores1[int(0.025 * len(sorted_scores1))]
	confidence_upper1 = sorted_scores1[int(0.975 * len(sorted_scores1))]
	
	confidence_lower2 = sorted_scores2[int(0.025 * len(sorted_scores2))]
	confidence_upper2 = sorted_scores2[int(0.975 * len(sorted_scores2))]
	
	confidence_lower_diff = sorted_scores_diff[int(0.025 * len(sorted_scores_diff))]
	confidence_upper_diff = sorted_scores_diff[int(0.975 * len(sorted_scores_diff))]
	
	return (sorted_scores1, confidence_lower1, confidence_upper1), (sorted_scores2, confidence_lower2, confidence_upper2), (sorted_scores_diff, confidence_lower_diff, confidence_upper_diff)

##########################################################################################################################
def process_results(results, class_label_mapping):
    y_true_numeric = np.array([class_label_mapping[label] for label in results['y_true']])
    y_prob_raw = results['y_prob_raw'].values
    y_pred = np.array([class_label_mapping[label] for label in results['y_pred']])
    
    return y_true_numeric, y_prob_raw, y_pred


def read_data(file_path):
    return pd.read_csv(file_path)



def plot_roc_and_metrics(results, class_label_mapping, result_type, data_source, Level_patch, model_type):
    class_label_mapping = {'background': 0, 'tissue': 1}
    y_true_numeric = np.array([class_label_mapping[label] for label in results['y_true']])
    y_prob_raw = results['y_prob_raw'].values
    y_pred = np.array([class_label_mapping[label] for label in results['y_pred']])
    
    auc_value = calculate_roc_and_plot(y_true_numeric, y_prob_raw)

    # Save the ROC plot
    save_path = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_roc_curve.png'
    plt.savefig(save_path)

    f1_score_value = f1_score(y_true_numeric, y_pred)
    recall_score_value = recall_score(y_true_numeric, y_pred)
    precision = precision_score(y_true_numeric, y_pred)

    return auc_value, f1_score_value, recall_score_value , precision
############################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          current_ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
 
    if normalize:
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Set zero sums to 1 to avoid division by zero
        cm2 = cm.astype('float') / row_sums[:, np.newaxis]

        cm_normalized = (cm2.astype('float') - np.amin(cm2)) / (np.amax(cm2)-np.amin(cm2))
    else:
        cm_normalized = (cm.astype('float') - np.amin(cm)) / (np.amax(cm)-np.amin(cm))

    ax = current_ax
    if normalize:
        im = ax.imshow(cm2, interpolation='nearest', cmap=cmap)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=20)  # Increase title font size

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, horizontalalignment="right", fontsize=18)  # Increase x-axis label font size
    ax.set_yticklabels(classes, rotation=45, fontsize=18)  # Increase y-axis label font size
    ax.set_ylim((len(classes)-0.5, -0.5))

    fmt = '.2%' if normalize else '.2%'
    thresh = 0.5
    text_font_size = 22  # Increase text font size
    cell_values = []  # List to store cell values for CSV

    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            cell_value = '{} ({:.2%})'.format(cm[i, j], cm2[i, j])
            cell_values.append(cell_value)
            ax.text(j, i, cell_value,
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=text_font_size,
                    color="white" if cm_normalized[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            cell_value = format(cm[i, j], fmt)
            cell_values.append(cell_value)
            ax.text(j, i, cell_value,
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=text_font_size,
                    color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_ylabel('Truth', fontsize=18)  # Increase y-axis label font size
    ax.set_xlabel('Predicted', fontsize=18)  # Increase x-axis label font size
    
    return cell_values


def plot_confusion_matrix_and_save(results, class_names, result_type, ext_data_source, data_source, Level_patch, model_type):
    label_index_arr = np.asarray(results['y_true'], dtype=str)
    pred_arr = np.asarray(results['y_pred'], dtype=str)
    conf_mat = confusion_matrix(label_index_arr, pred_arr)

    fig, ax = plt.subplots(figsize=(10, 10))
    cell_values = plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, current_ax=ax)
    fig.subplots_adjust(left=0.15, bottom=0.28, right=0.94, top=0.99, wspace=0.2, hspace=0.20)

    save_path = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{data_source}_{Level_patch}_{ext_data_source}_results_cm.png'
    plt.savefig(save_path)

    save_path_csv = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{data_source}_{Level_patch}_{ext_data_source}_cell_values.csv'
    df = pd.DataFrame(np.array(cell_values).reshape(conf_mat.shape), columns=class_names, index=class_names)
    df.to_csv(save_path_csv, index=True)  # Save cell values as CSV

#########################################################################
def save_wrongly_predicted_patches(results, result_type, ext_data_source, data_source, Level_patch, model_type):
    background_classes = ['background']
    tissue_classes = ['tissue']
    wrongly_predicted = results[results['y_true'] != results['y_pred']]
    wrongly_predicted.to_csv(f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{ext_data_source}_Wrong_results.csv', index=False)

    sorted_wrongly_predicted = wrongly_predicted.sort_values(by='y_prob_raw', ascending=False)
    sorted_wrongly_predicted_decending = wrongly_predicted.sort_values(by='y_prob_raw', ascending=True) #NOT SURE ABOUT IT LET"S SEE

    # Plot top 30 tissues predicted as background
    top_wrongly_tissues_predicted_as_background= sorted_wrongly_predicted_decending[sorted_wrongly_predicted_decending['y_pred'].isin((background_classes))].head(30)
    top_wrongly_backgrounds_predicted_as_tissue = sorted_wrongly_predicted[sorted_wrongly_predicted['y_pred'].isin(tissue_classes)].head(30)

    top_wrongly_tissues_predicted_as_background['patch_id'] = top_wrongly_tissues_predicted_as_background['patch_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    top_wrongly_backgrounds_predicted_as_tissue['patch_id'] = top_wrongly_backgrounds_predicted_as_tissue['patch_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    
    top_wrongly_backgrounds_predicted_as_tissue_file_path = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{ext_data_source}_top_wrongly_backgrounds_predicted_as_tissue_info.csv'
    top_wrongly_tissues_predicted_as_background_file_path = f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{ext_data_source}_top_wrongly_tissues_predicted_as_background_info.csv'
    top_wrongly_backgrounds_predicted_as_tissue[['patch_path', 'y_true', 'y_pred', 'y_prob_raw']].to_csv(top_wrongly_backgrounds_predicted_as_tissue_file_path, index=False)
    top_wrongly_tissues_predicted_as_background[['patch_path', 'y_true', 'y_pred', 'y_prob_raw']].to_csv(top_wrongly_tissues_predicted_as_background_file_path, index=False)
    
    fig, axs = plt.subplots(5, 6, figsize=(15, 10))
    axs = axs.flatten()
    for i, (_, row) in enumerate(top_wrongly_tissues_predicted_as_background.iterrows()):
        img = mpimg.imread(row['patch_path'])
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'ID: {row["patch_id"]}')  # Set individual titles
        # axs[i].text(0.5, -0.15, f'ID: {row["patch_id"]}', transform=axs[i].transAxes, horizontalalignment='center')

    fig.suptitle(f'Top 30 {result_type}_{ext_data_source} Predicted as backgroun', x=0.5, y=1.05, ha='center', fontsize=14)
    plt.savefig(f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{ext_data_source}_top_Predicted_as_backgroun.png')

    # Create a new set of axes for the second set of images
    fig, axs = plt.subplots(5, 6, figsize=(15, 10))
    axs = axs.flatten()
    for i, (_, row) in enumerate(top_wrongly_backgrounds_predicted_as_tissue.iterrows()):
        img = mpimg.imread(row['patch_path'])
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'ID: {row["patch_id"]}')  # Set individual titles
        # axs[i].text(0.5, -0.15, f'ID: {row["patch_id"]}', transform=axs[i].transAxes, horizontalalignment='center')

    fig.suptitle(f'Top 30 {result_type} Predicted as tissue', x=0.5, y=1.05, ha='center', fontsize=14)
    plt.savefig(f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_{ext_data_source}_top_Predicted_as_tissue.png')

###############################################################################

def calculate_confidence_intervals(y_true_numeric, y_prob_raw, result_type, ext_data_source, data_source, Level_patch, model_type):
    confidence_filename = f'./results/{data_source}/val_confidence_intervals.csv' if result_type == 'val'  else f'./results/{data_source}/test_confidence_intervals.csv' if result_type == 'test' else f'./results/{data_source}/{ext_data_source}_external_val_confidence_intervals.csv'

    try:
        confidence_df = pd.read_csv(confidence_filename)
    except FileNotFoundError:
        confidence_df = pd.DataFrame(columns=['#dataset', '#AUC', '#upper', '#lower'])

    bootstrapped_scores, confidence_lower, confidence_upper = BootStrap(y_true_numeric, y_prob_raw, n_bootstraps=1000)
    if ext_data_source == '_' :
        result = pd.DataFrame({'#dataset': [f'{data_source}_{Level_patch}'], '#AUC': [score_fnc(y_true_numeric, y_prob_raw)],
                            '#upper': [confidence_upper], '#lower': [confidence_lower]})
    else :
        result = pd.DataFrame({'#dataset': [f'{ext_data_source}_{Level_patch}'], '#AUC': [score_fnc(y_true_numeric, y_prob_raw)],
                           '#upper': [confidence_upper], '#lower': [confidence_lower]})
    confidence_df = pd.concat([confidence_df, result])
    confidence_df.to_csv(confidence_filename, index=False)

    plt.figure(figsize=(8, 6))
    plt.hist(bootstrapped_scores, bins=30, alpha=0.7, color='orange', label='Bootstrap Scores -{result_type}')
    plt.axvline(x=confidence_lower, color='red', linestyle='--', label='Lower Confidence Bound')
    plt.axvline(x=confidence_upper, color='green', linestyle='--', label='Upper Confidence Bound')
    # Annotate AUC value on the plot
    plt.text(0.5, 0.9, f'mean: {np.mean(bootstrapped_scores):.3f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # Annotate Lower Confidence Bound value on the plot
    plt.text(0.5, 0.85, f'Lower Bound: {confidence_lower:.3f}', color='red', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # Annotate Upper Confidence Bound value on the plot
    plt.text(0.5, 0.80, f'Upper Bound: {confidence_upper:.3f}', color='green', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.xlabel('AUC Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Bootstrap Results - Validation Data')
    # Save the plot
    plt.savefig(f'./results/{data_source}/{Level_patch}/train_test_summary/{model_type}/{result_type}_bootstrap_plot.png')

##########################################################################################################################
