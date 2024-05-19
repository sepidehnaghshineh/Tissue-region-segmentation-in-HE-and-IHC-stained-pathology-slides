import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# List of file paths
file_paths = [
    './Segmentation/segmentation_results/LeNet5/resolution_4/source_name/OTSU_scores.txt',
    './Segmentation/segmentation_results/LeNet5/resolution_4/source_name/scores.txt'
]

# Create a list to store parsed data
all_parsed_data = []

# Loop through each file path
for file_path in file_paths:
    # Read the data from the file
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        data = file.read()

    # Split the data by lines
    lines = data.split('\n')

    # Parse the data into a list of dictionaries
    parsed_data = []
    for line in lines:
        values = line.split(',')
        if len(values) == 3:
            parsed_data.append({
                'slide_name': values[0],
                'jaccard_score': float(values[1]),
                'dice_coef': float(values[2])
            })

    # Append parsed data to the list
    all_parsed_data.append(parsed_data)

# Create a DataFrame for each model's data
dfs = [pd.DataFrame(data) for data in all_parsed_data]

# Create a single figure for box plots
plt.figure(figsize=(14, 7))

# Function to calculate outliers and print them
def find_and_print_outliers(df, score_type, model_name):
    Q1 = df[score_type].quantile(0.25)
    Q3 = df[score_type].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[score_type] < lower_bound) | (df[score_type] > upper_bound)]
    print(f"Outliers for {model_name} - {score_type}:")
    print(outliers[['slide_name', score_type]])
    return outliers

# Dictionary to store outliers
outliers_data = {'Model': [], 'Score_Type': [], 'Slide_Name': [], 'Score': []}

# Box plot for Jaccard scores
plt.subplot(1, 2, 1)
plt.title('Jaccard Scores Comparison')
plt.ylabel('Jaccard Score')
boxplot_jaccard = plt.boxplot([df['jaccard_score'] for df in dfs], labels=['OTSU', 'segmentation_model'])
plt.xticks(rotation=20)
plt.yticks(np.arange(0, 1.1, 0.1))

# Print and annotate outliers for Jaccard scores
for i, (df, label) in enumerate(zip(dfs, ['OTSU', 'segmentation_model'])):
    outliers = find_and_print_outliers(df, 'jaccard_score', label)
    for _, outlier in outliers.iterrows():
        plt.annotate(outlier['slide_name'], xy=(i + 1, outlier['jaccard_score']),
                     xytext=(i + 1.2, outlier['jaccard_score']),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        outliers_data['Model'].append(label)
        outliers_data['Score_Type'].append('Jaccard')
        outliers_data['Slide_Name'].append(outlier['slide_name'])
        outliers_data['Score'].append(outlier['jaccard_score'])

# Box plot for Dice coefficients
plt.subplot(1, 2, 2)
plt.title('Dice Coefficients Comparison')
plt.ylabel('Dice Coefficient')
boxplot_dice = plt.boxplot([df['dice_coef'] for df in dfs], labels=['OTSU', 'segmentation_model'])
plt.xticks(rotation=20)
plt.yticks(np.arange(0, 1.1, 0.1))

# Print and annotate outliers for Dice coefficients
for i, (df, label) in enumerate(zip(dfs, ['OTSU', 'segmentation_model'])):
    outliers = find_and_print_outliers(df, 'dice_coef', label)
    for _, outlier in outliers.iterrows():
        plt.annotate(outlier['slide_name'], xy=(i + 1, outlier['dice_coef']),
                     xytext=(i + 1.2, outlier['dice_coef']),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        outliers_data['Model'].append(label)
        outliers_data['Score_Type'].append('Dice')
        outliers_data['Slide_Name'].append(outlier['slide_name'])
        outliers_data['Score'].append(outlier['dice_coef'])

plt.tight_layout()

# Save the upper, lower boundaries, and means to a CSV file
boundaries_data = pd.DataFrame({
    'Model': ['OTSU', 'segmentation_model'],
    'Jaccard_Upper': [item.get_ydata()[0] for item in boxplot_jaccard['caps'][1::2]],
    'Jaccard_Lower': [item.get_ydata()[0] for item in boxplot_jaccard['caps'][0::2]],
    'Jaccard_Mean': [np.mean(df['jaccard_score']) for df in dfs],
    'Dice_Upper': [item.get_ydata()[0] for item in boxplot_dice['caps'][1::2]],
    'Dice_Lower': [item.get_ydata()[0] for item in boxplot_dice['caps'][0::2]],
    'Dice_Mean': [np.mean(df['dice_coef']) for df in dfs]
})

boundaries_data.to_csv('./Segmentation/segmentation_results/boundaries_data.csv', index=False)

# Save the outliers data to a CSV file
outliers_df = pd.DataFrame(outliers_data)
outliers_df.to_csv('./Segmentation/segmentation_results/outliers_data.csv', index=False)

# Save the plot as a PDF
plt.savefig('./Segmentation/segmentation_results/ALL_data_comparison_plot.png')

plt.show()
