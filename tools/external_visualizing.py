import ast
import sys
sys.path.append('./')
import argparse
import numpy as np
import matplotlib.image as mpimg
import itertools
from tools.data_visualising import  plt_charts, plot_roc_and_metrics , plot_confusion_matrix_and_save , save_wrongly_predicted_patches, calculate_confidence_intervals , process_results
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score , roc_curve , auc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import ast
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, auc, recall_score , precision_score

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

def plot_roc_and_metrics(results, class_label_mapping, result_type, data_source):
    class_label_mapping = {'background': 0, 'tissue': 1}
    y_true_numeric = np.array([class_label_mapping[label] for label in results['y_true']])
    y_prob_raw = results['y_prob_raw'].values
    y_pred = np.array([class_label_mapping[label] for label in results['y_pred']])
    
    auc_value = calculate_roc_and_plot(y_true_numeric, y_prob_raw)

    # Save the ROC plot
    save_path = f'./results/external/{data_source}/{result_type}_roc_curve.pdf'
    plt.savefig(save_path, dpi=500, format ='pdf')

    f1_score_value = f1_score(y_true_numeric, y_pred)
    recall_score_value = recall_score(y_true_numeric, y_pred)
    precision = precision_score(y_true_numeric, y_pred)

    return auc_value, f1_score_value, recall_score_value , precision



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
        ax.set_title(title, fontweight="bold", fontsize=18)  # Increase title font size

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, horizontalalignment="right", fontsize=14)  # Increase x-axis label font size
    ax.set_yticklabels(classes, rotation=45, fontsize=14)  # Increase y-axis label font size
    ax.set_ylim((len(classes)-0.5, -0.5))

    fmt = '.2%' if normalize else '.2%'
    thresh = 0.5
    text_font_size = 18  # Increase text font size
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

    ax.set_ylabel('Truth', fontsize=16)  # Increase y-axis label font size
    ax.set_xlabel('Predicted', fontsize=16)  # Increase x-axis label font size
    
    return cell_values

def plot_confusion_matrix_and_save(results, class_names, ext_data_source, data_source):
    label_index_arr = np.asarray(results['y_true'], dtype=str)
    pred_arr = np.asarray(results['y_pred'], dtype=str)
    conf_mat = confusion_matrix(label_index_arr, pred_arr)

    fig, ax = plt.subplots(figsize=(10, 10))
    cell_values = plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, current_ax=ax)
    fig.subplots_adjust(left=0.15, bottom=0.28, right=0.94, top=0.99, wspace=0.2, hspace=0.20)

    save_path_pdf = f'./results/external/{data_source}/{ext_data_source}_results_cm.pdf'
    plt.savefig(save_path_pdf, dpi=500, format ='pdf')  # Save figure as PDF

    save_path_csv = f'./results/external/{data_source}/{ext_data_source}_cell_values.csv'
    df = pd.DataFrame(np.array(cell_values).reshape(conf_mat.shape), columns=class_names, index=class_names)
    df.to_csv(save_path_csv, index=True)  # Save cell values as CSV






def save_wrongly_predicted_patches(results, ext_data_source, data_source):
    background_classes = ['background']
    tissue_classes = ['tissue']
    wrongly_predicted = results[results['y_true'] != results['y_pred']]
    wrongly_predicted.to_csv(f'./results/external/{data_source}/{ext_data_source}_Wrong_results.csv', index=False)

    sorted_wrongly_predicted = wrongly_predicted.sort_values(by='y_prob_raw', ascending=False)
    sorted_wrongly_predicted_decending = wrongly_predicted.sort_values(by='y_prob_raw', ascending=True) #NOT SURE ABOUT IT LET"S SEE

    # Plot top 30 tissues predicted as background
    top_wrongly_tissues_predicted_as_background= sorted_wrongly_predicted_decending[sorted_wrongly_predicted_decending['y_pred'].isin((background_classes))].head(30)
    top_wrongly_backgrounds_predicted_as_tissue = sorted_wrongly_predicted[sorted_wrongly_predicted['y_pred'].isin(tissue_classes)].head(30)

    top_wrongly_tissues_predicted_as_background['patch_id'] = top_wrongly_tissues_predicted_as_background['patch_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    top_wrongly_backgrounds_predicted_as_tissue['patch_id'] = top_wrongly_backgrounds_predicted_as_tissue['patch_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    
    top_wrongly_backgrounds_predicted_as_tissue_file_path = f'./results/external/{data_source}/{ext_data_source}_top_wrongly_backgrounds_predicted_as_tissue_info.csv'
    top_wrongly_tissues_predicted_as_background_file_path = f'./results/external/{data_source}/{ext_data_source}_top_wrongly_tissues_predicted_as_background_info.csv'
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

    fig.suptitle(f'Top 30 {ext_data_source} Predicted as backgroun', x=0.5, y=1.05, ha='center', fontsize=14)
    plt.savefig(f'./results/external/{data_source}/{ext_data_source}_top_Predicted_as_backgroun.pdf', dpi=500, format ='pdf')

    # Create a new set of axes for the second set of images
    fig, axs = plt.subplots(5, 6, figsize=(15, 10))
    axs = axs.flatten()
    for i, (_, row) in enumerate(top_wrongly_backgrounds_predicted_as_tissue.iterrows()):
        img = mpimg.imread(row['patch_path'])
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'ID: {row["patch_id"]}')  # Set individual titles
        # axs[i].text(0.5, -0.15, f'ID: {row["patch_id"]}', transform=axs[i].transAxes, horizontalalignment='center')

    fig.suptitle(f'Top 30 {ext_data_source} Predicted as tissue', x=0.5, y=1.05, ha='center', fontsize=14)
    plt.savefig(f'./results/external/{data_source}/{ext_data_source}_top_Predicted_as_tissue.pdf', dpi=500, format ='pdf')