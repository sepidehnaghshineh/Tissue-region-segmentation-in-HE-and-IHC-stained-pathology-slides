import os
import fnmatch
import argparse

"""

The Outputs will be created in the same directory as the input directory as a folder named "Crops_info"

"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./Dataset_Prep/data_preparation_results/Cropped_slides', help="Directory of the \"Cropped_slides\" folder")
FLAGS = parser.parse_args()

input_dir = FLAGS.input_dir

if not os.path.exists(input_dir):
    print("Specified directory does not exist")
    exit()

print("Saving source info...")
allpatches_dict = {}
#iterate over every text file in the directory X that starts with allpatches
for root, dirs, files in os.walk(input_dir):
    for file in fnmatch.filter(files, ('allpatches'+ '*.txt')):
        
        allpatches_txt_path = os.path.join(root, file)
        try:
            level_size_info = file.split('.')[0].split("__")[1]
        except:
            level_size_info = file.split('.')[0].split("_")[1:-1]
        level = int(level_size_info.split('_')[1])
        size = int(level_size_info.split('_')[3])
        num_tissue_allpatches = 0
        num_background_allpatches = 0
        num_total_allpatches = 0
        source = "unknown"
        if "source_name" in root:
            source = "source_name"
        with open(allpatches_txt_path, "r") as allpatches_txt:
            _ = allpatches_txt.readline() #skip the header
            for line in allpatches_txt.readlines():
                line = line.split(",")
                label = line[3]
                if label == "tissue":
                    num_tissue_allpatches += 1
                elif label == "background":
                    num_background_allpatches += 1
                num_total_allpatches += 1
        
        allpatches_dict[f"{source}_Level{level}_Size{size}"] = [num_tissue_allpatches, num_background_allpatches, num_total_allpatches]
                

Crops_info_directory = os.path.join(input_dir, os.path.pardir, "Crops_info")


os.makedirs(Crops_info_directory, exist_ok=True)
    
with open(os.path.join(Crops_info_directory, "source_info.txt"), "w") as source_info_txt:
    source_info_txt.write("Source_Level_Size, Tiss_Num, Tiss_(%), Bg_Num, Bg_(%) , Total_Num\n")
    for key, value in allpatches_dict.items():
        source_info_txt.write(f"{key}, {value[0]}, %{round(value[0]/value[2]*100,2)}, {value[1]}, %{round(value[1]/value[2]*100,2)}, {value[2]}\n")

print("Source info saved successfully")
