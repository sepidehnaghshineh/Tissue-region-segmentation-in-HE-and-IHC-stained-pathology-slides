import os
import sys
import argparse
import pandas as pd
sys.path.append(os.getcwd())
from tools.CroppingTools import generate_allpatches

parser = argparse.ArgumentParser(description='Cropping and Saving')

parser.add_argument('--input_dir', type=str, default=r"./Tissue_region_Segmentation/Dataset_Prep/data_preparation_results/Cropped_slides/source_name/test", help='info.txt file path')

FLAGS = parser.parse_args()

input_dir = FLAGS.input_dir

generate_allpatches(input_dir)