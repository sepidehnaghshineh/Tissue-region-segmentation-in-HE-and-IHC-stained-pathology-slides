import os
import numpy as np 

def read_patch_paths(dataset_dir):

    # Read the text file and parse patch information
    patch_info = []
    with open(dataset_dir, 'r') as file:
        for line in file:
            patient_id,slide_id,patch_id,patch_label,patch_path,tissue_ratio,bg_ratio,base_coords,mask_coords,mask_lvl = line.strip().split(',')
            patch_info.append((patient_id.strip(),slide_id.strip(), patch_id.strip(), patch_label.strip(), patch_path.strip(), tissue_ratio.strip(), bg_ratio.strip(), base_coords.strip(),  mask_coords.strip(), mask_lvl.strip()))

            # patient_id, slide_id, patch_id, patch_label, patch_path, tissue_ratio, bg_ratio ,  mask_y, mask_x = line.strip().split(',')
            # patch_info.append((patient_id.strip(),slide_id.strip(), patch_id.strip(), patch_label.strip(), patch_path.strip(), tissue_ratio.strip(), bg_ratio.strip(), mask_y.strip(),  mask_x.strip()))

    # Extract unique patch labels excluding 'patch_label'
    class_names = np.unique([patch[3] for patch in patch_info if patch[3] != 'patch_label'])

    # Group patch paths based on patch label and patient ID
    patch_files = {}
    for patch_label in class_names:
        patch_files[patch_label] = [os.path.join(patch[4]) for patch in patch_info if patch[3] == patch_label]

    patch_file_list = []
    patch_label_list = []

    # Assign numerical labels to patch labels
    patch_label_to_num = {label: i for i, label in enumerate(class_names)}

    for patch in patch_info:
        if patch[3] != 'patch_label':  # Exclude 'patch_label' from the list
            patch_file_list.append(os.path.join(patch[4]))
            patch_label_list.append(patch_label_to_num[patch[3]])

    return patch_file_list, patch_label_list, class_names ,tissue_ratio, bg_ratio, base_coords,mask_coords,mask_lvl



def dataset_info(dataset_dir):
    
    patch_file_list, patch_label_list, class_names, tissue_ratio, bg_ratio, base_coords,mask_coords,mask_lvl= read_patch_paths(dataset_dir)
    
    class_counts = [patch_label_list.count(i) for i in range(len(class_names))]

    return patch_file_list, patch_label_list, class_counts, class_names ,tissue_ratio, bg_ratio,base_coords,mask_coords,mask_lvl


