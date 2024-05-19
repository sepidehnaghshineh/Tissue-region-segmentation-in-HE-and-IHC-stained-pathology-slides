import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
from scipy.stats import mstats
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_source', default='source_name', help='the source that you have got the WSIs', dest='data_source')
parser.add_argument('--model_type', default='LeNet5', help='the used CNN model', dest='model_type')
parser.add_argument('--model_mode', default='test', help='test or validation', dest='model_mode')
FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

def create_and_save_box_plot(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a box plot
    plt.figure(figsize=(12, 8))
    box_data = []

    # Define colors for each dataset
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightpink', 'lightgray']
 
    for i, (index, row) in enumerate(df.iterrows()):
        lower = row['#lower']
        upper = row['#upper']
        auc = row['#AUC']
        dataset_color = colors[i % len(colors)]  # Use modulo to cycle through colors
        box_data.append([lower, upper])

        # Use a different color for each dataset
        plt.boxplot([lower, upper], positions=[i + 1], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=dataset_color, color='black'),
                    capprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    medianprops=dict(color='black'))
        
        # Draw grid lines touching the box edges with the same color
        plt.axhline(lower, color=dataset_color, linestyle='--', linewidth=0.5)
        plt.axhline(upper, color=dataset_color, linestyle='--', linewidth=0.5)

    # Set labels and title
    plt.xticks(range(1, len(df) + 1), df['#dataset'], rotation=10, ha='right', fontsize=18, fontweight='bold')
    plt.ylabel('Confidence Intervals of AUC')
    plt.title(f'BoxPlot of {FLAGS.data_source}_{file_name}_of AUC')  # Set the title based on file name

    # Generate a file name based on the input file path
    save_name = os.path.splitext(os.path.basename(file_path))[0] + '_boxplot.png'
    save_name = os.path.splitext(os.path.basename(file_path))[0] + '_boxplot.png'
    save_path = os.path.join(os.path.dirname(file_path), save_name)

    # Adjust white space around the plot
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    # Save the plot
    plt.savefig(save_path)
    print(f"Box plot saved to: {save_path}")

# Load the CSV file into a DataFrame
val_file_path = ('./results/{}/val_confidence_intervals.csv'.format(FLAGS.data_source))
test_file_path= ('./results/{}/test_confidence_intervals.csv'.format(FLAGS.data_source))

create_and_save_box_plot(val_file_path)
create_and_save_box_plot(test_file_path)