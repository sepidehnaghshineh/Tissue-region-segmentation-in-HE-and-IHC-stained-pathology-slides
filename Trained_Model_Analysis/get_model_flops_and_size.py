import argparse
import os
import torch
import sys
sys.path.append('./')
from fvcore.nn import FlopCountAnalysis, flop_count_table
import pandas as pd

# Import your LeNet models here
from tools.LeNet5_different_inputSizes import LeNet5_128, LeNet5_64, LeNet5_32, LeNet5_16

parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--input_size', type=int, default=128, help='Input size for LeNet models (e.g., 128, 64, 32, 16)', dest='input_size')

FLAGS = parser.parse_args()

print('##### input_size:{} #####'.format(FLAGS.input_size))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create a DataFrame to store the model names and FLOPs
model_df = pd.DataFrame(columns=['Model', 'Total FLOPs', 'FLOP Table'])

# Function to select the appropriate model based on input size
def select_model(input_size):
    if input_size == 128:
        return LeNet5_128()
    elif input_size == 64:
        return LeNet5_64()
    elif input_size == 32:
        return LeNet5_32()
    elif input_size == 16:
        return LeNet5_16()
    else:
        raise ValueError("Unsupported input size")

# Loop through the input sizes and generate models
for size in [128, 64, 32, 16]:
    model_name = f'LeNet5_{size}'
    
    # Get the LeNet model for the current input size
    model = select_model(size)

    # Move the model to the device
    model.to(device)

    # Generate a random input tensor for flop count analysis
    x = torch.rand(1, 3, size, size, dtype=torch.float32).to(device)

    # Get the flop count
    flops = FlopCountAnalysis(model, x)
    total_flops = flops.total()

    # Save information to the DataFrame
    model_df = model_df.append({'Model': model_name, 'Total FLOPs': total_flops, 'FLOP Table': flop_count_table(flops)}, ignore_index=True)

# Print the DataFrame
print(model_df)

# Specify the directory to save the CSV file
save_directory = './results/'
# Save the model information to a CSV file
model_df.to_csv(os.path.join(save_directory,'model_flops.csv'), index=False)

