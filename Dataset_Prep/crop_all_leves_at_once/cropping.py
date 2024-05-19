import os
import sys
import argparse
import pandas as pd
import argparse
sys.path.append(os.getcwd())
sys.path.append('./')
from tools.CroppingTools import Croppable, crop_and_save

# if you want to run multiple instances of the same code, you can use this variables for naming shortcuts #
dtype: str = 'train'  # 'train', 'test', 'val', 'external'
dsource: str = 'source_name'  # 'source_name', 'external_source_name'
code_num: int|str = "" # for running multiple code instances at the same time
###########################################################################################################

if code_num == "" or None:
    pass
else:
    code_num = "_" + str(code_num)




parser = argparse.ArgumentParser(description='Cropping and Saving')
parser.add_argument('--input_dir', type=str, default=f"./Dataset_Prep/data_preparation_results/Tissue_Masks/{dsource}_masks/{dtype}_tissue_mask_info_{code_num}.txt", help='info.txt file path')
parser.add_argument('--input_dir', type=str, default=f"./Dataset_Prep/data_preparation_results/Tissue_Masks/source_name_masks/test_tissue_mask_info.txt", help='info.txt file path')
parser.add_argument('--out_dir', type=str, default=r"./Dataset_Prep/data_preparation_results/", help='output directory')
parser.add_argument('--patch_level',nargs="+" ,type=int, default=[4], help='The level of the patches that will be cropped.')
parser.add_argument('--patch_size',nargs="+" ,type=int, default=[128], help='The size of the patches for highest level.')

parser.add_argument('--seperate_bg_mask', type=bool, default=False, help='Use a seperate mask for background')
parser.add_argument('--seperate_input_dir', type=str, default=f"./Dataset_Prep/data_preparation_results/Tissue_Masks/{dsource}_masks/{dtype}_dilated_tissue_mask_info_{code_num}.txt", help='info.txt file path for sperate background mask')
FLAGS = parser.parse_args()

input_dir = FLAGS.input_dir
out_dir = FLAGS.out_dir
levels = FLAGS.patch_level
sizes = FLAGS.patch_size
sep_bg_mask = FLAGS.seperate_bg_mask
sep_input_dir = FLAGS.seperate_input_dir

df = pd.read_csv(input_dir)

if sep_bg_mask:
    if sep_input_dir == "":
        raise Exception("You have to specify a seperate input directory for the additional info.txt")
    sep_df = pd.read_csv(sep_input_dir)
else:
    sep_df = df.copy()
    
train_test_val = "not_specified"
if "train" in os.path.basename(input_dir):
    train_test_val = "train"
elif "test" in os.path.basename(input_dir):
    train_test_val = "test"
elif "val" in os.path.basename(input_dir):
    train_test_val = "val"
elif "external" in os.path.basename(input_dir):
    train_test_val = "external"



#load class and slide
for i in range(len(df)):
    croppable = Croppable(df.iloc[i], sep_df.iloc[i])
    crop_and_save(croppable, levels, sizes, out_dir, train_test_val)