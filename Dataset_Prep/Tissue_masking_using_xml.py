import os
from lxml import etree
import openslide
from PIL import Image
import numpy as np
import cv2
import argparse
import math


def get_eroded_mask(mask_path, kernel_size=2, iterations=1, inverted=True):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if inverted:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    if inverted:
        eroded_mask = cv2.bitwise_not(eroded_mask)
    return eroded_mask

def get_dilated_mask(mask_path, kernel_size=2, iterations=1, inverted=True):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if inverted:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    if inverted:
        dilated_mask = cv2.bitwise_not(dilated_mask)
    return dilated_mask


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='The directory of the source folder that contains the slides.')
parser.add_argument('--output_dir', type=str, default='./Dataset_Prep/data_preparation_results/', help='The directory of the output folder that will contain the masks.')

# Below are the optional arguments (if you want to use erosion and dilation).
parser.add_argument('--erosion_size', type=int, default=5, help='The size of the erosion kernel.')
parser.add_argument('--erosion_iterations', type=int, default=0, help='The number of iterations for erosion.')
parser.add_argument('--dilation_size' ,type=int, default=5, help='The size of the dilation kernel.')
parser.add_argument('--dilation_iterations', type=int, default=0, help='The number of iterations for dilation.')
FLAGS = parser.parse_args()

folder_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
erosion_kernel_size = FLAGS.erosion_size
erosion_iterations = FLAGS.erosion_iterations

slidelist = []
data_type = ''
print('started')
base_name = os.path.basename(folder_dir)
for root, dirs, files in os.walk(folder_dir):

    f1, f2 = False, False
    data_type = ''
    for file in files:
        if file.startswith('._'):
            continue
        f1, f2 = False, False
        if file.rsplit('.',1)[0] + '.xml' in os.listdir(root):
            f1 = True
        else:
            continue
        root = root.replace('\\','/')
        root_as_list = root.split('/')
        file_index = root_as_list.index(base_name)
        root_as_list = root_as_list[file_index:]
        for i in root_as_list:
            if i.lower() == "train" or i.lower() == "training":
                data_type = "train_"
            if i.lower() == "test" or i.lower() == "testing":
                data_type = "test_"
            if i.lower() == "val" or i.lower() == "validation" or i.lower() == 'validating':
                data_type = "val_"
            if i.lower() == "ext" or i.lower() == "external":
                data_type = "external_"
                
        if file.endswith('.svs') or  file.endswith('.tiff') or file.endswith('.tif') or file.endswith('.mrxs') or file.endswith('.ndpi') or file.endswith('.ndpi') and file.startswith('source_name'):
            f2 = True
            source = 'source_name' #put the source name where you get the WSI from here. 
            patient_id = root_as_list[-1]
            slide_id = file.split('.')[0]
        
        if f2 & f1:
            slide_path = os.path.join(root, file)
            xml_path = os.path.join(root, file.rsplit('.',1)[0] + '.xml')
            slidelist.append([slide_path, xml_path, source, patient_id, slide_id, data_type])
            # print(data_type)

data_types_list = ["", "train_", "test_", "val_", "external_"]
sources_list = ["source_name"]
Data_stats_path_created_bool_dict = dict() # ["", "train", "test", "val" , "external"]
txt_file_path_created_bool_list = dict() # ["", "train", "test", "val" , "external"]
for source in sources_list:
    Data_stats_path_created_bool_dict[source] = [False, False, False, False, False]
    txt_file_path_created_bool_list[source] = [False, False, False, False, False]


for slide_path, xml_file_path, source, patient_id, slide_id, data_type in slidelist:

    # Load the XML file
    tree = etree.parse(xml_file_path)
    root = tree.getroot()

    # Load the slide image
    slide = openslide.OpenSlide(slide_path)
    
    for i in range(len(slide.level_dimensions)):
        if slide.level_dimensions[i][0] > 700 and slide.level_dimensions[i][1] > 700:
            best_level_dimensions_index = i
            
    # Calculate the target downsample
    target_downsample = math.log(int(round(slide.level_downsamples[best_level_dimensions_index])), 2)

    # Set the resolution of the slide image
    try:
        val_x = float(slide.properties.get('openslide.mpp-x'))
    except:
        try:
            res_type = slide.properties.get("tiff.ResolutionUnit")
            if res_type == "centimeter":
                numerator = 10000
            elif res_type == "inch":
                numerator = 25400
            val_x = numerator / float(slide.properties.get("tiff.XResolution"))
        except:
            print('Unknown Val_x')
            continue
        
    if val_x < 0.3:  # resolution:0.25um/pixel
        current_res = 0.25
    elif val_x < 0.6:  # resolution:0.5um/pixel
        current_res = 0.5

    im_size = slide.level_dimensions[best_level_dimensions_index]
    target_res = (2 ** (target_downsample))
    normalized_level = target_downsample + (current_res * 4) - 1
    real_length = target_res * current_res  # real length in micron
    
    # Initialize the tissue mask and RGB mask
    mask_width, mask_height = im_size
    tissue_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # Initialize the RGB mask
    rgb_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    roi_information = [0, 0, mask_width, mask_height] # [x, y, height, width]
    
    # Loop over all Annotation tags in the XML file
    annotations = tree.findall('.//Annotation')
    for ann in annotations:
        part_of_group = ann.attrib.get('PartOfGroup')
        if part_of_group == "tissue":
            coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = int(float(co.attrib.get('X').replace(',', '.')) / target_res)
                    Y = int(float(co.attrib.get('Y').replace(',', '.')) / target_res)
                    coordinates.append([X, Y])

            vertices = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(tissue_mask, [vertices], color=255)

        elif part_of_group == "bg":
            parent_annotation = ann.getparent()
            parent_coordinates = []
            for child_ann in parent_annotation.findall('Annotation'):
                if child_ann.attrib.get('PartOfGroup') != "bg":
                    continue

                child_coordinates = []
                for co in child_ann.iter():
                    if co.tag == 'Coordinate':
                        X = int(float(co.attrib.get('X').replace(',', '.')) / target_res)
                        Y = int(float(co.attrib.get('Y').replace(',', '.')) / target_res)
                        child_coordinates.append([X, Y])

                child_vertices = np.array(child_coordinates, dtype=np.int32)
                cv2.fillPoly(tissue_mask, [child_vertices], color=0)
                
        elif part_of_group == "roi":
            roi_coordinates = []
            for co in ann.iter():
                if co.tag == 'Coordinate':
                    X = max(int(float(co.attrib.get('X').replace(',', '.')) / target_res), 0)
                    Y = max(int(float(co.attrib.get('Y').replace(',', '.')) / target_res), 0)
                    roi_coordinates.append([X, Y])
                    
            top_left = roi_coordinates[0]
            bottom_right = roi_coordinates[2]
            x1, y1 = top_left
            x2, y2 = bottom_right
            width, height = x2 - x1, y2 - y1

            roi_information = [x1, y1, min(mask_width - x1, width), min(mask_height - y1,height)] # [x, y, width, height]
    roi_str = "-".join([str(roi_information[0]), str(roi_information[1]), str(roi_information[2]), str(roi_information[3])])
    
    # Invert the tissue mask
    inverted_tissue_mask = cv2.bitwise_not(tissue_mask)
    
    # Calculate the area of the tissue mask
    Mask_area_px = np.count_nonzero(tissue_mask)
    im_area_px = mask_width * mask_height
    Mask_area_ratio = Mask_area_px / im_area_px
    
    # Convert tissue mask to RGB mask using original RGB colors
    slide_rgb = slide.read_region((0, 0), best_level_dimensions_index, im_size)
    slide_rgb = np.array(slide_rgb.convert("RGB"))
    rgb_mask[tissue_mask == 255] = slide_rgb[tissue_mask == 255]


    output_directory = os.path.join(output_dir, "Tissue_Masks", source + '_masks')
    crop_info_directory = os.path.join(output_dir, "Crops_info")
    
    mask_output_dir = os.path.join(output_directory, f'{data_type}binary-mask')
    rgb_mask_output_dir = os.path.join(output_directory, f'{data_type}RGB-mask')
    
    if FLAGS.erosion_size != 0 and FLAGS.erosion_iterations != 0:
        eroded_mask_output_dir = os.path.join(output_directory, f'{data_type}eroded-mask')
        os.makedirs(eroded_mask_output_dir, exist_ok=True)
        eroded_txt_file_path = os.path.join(output_directory, data_type + 'eroded_tissue_mask_info' + '.txt')

    if FLAGS.dilation_size != 0 and FLAGS.dilation_iterations != 0:
        dilated_mask_output_dir = os.path.join(output_directory, f'{data_type}dilated-mask')
        os.makedirs(dilated_mask_output_dir, exist_ok=True)
        dilated_txt_file_path = os.path.join(output_directory, data_type + 'dilated_tissue_mask_info' + '.txt')

    # Create the output directories if they do not exist
    os.makedirs(mask_output_dir, exist_ok=True)
    os.makedirs(rgb_mask_output_dir, exist_ok=True)
    os.makedirs(crop_info_directory, exist_ok=True)


    txt_file_path = os.path.join(output_directory, data_type + 'tissue_mask_info' + '.txt')
    Data_stats_path = os.path.join(crop_info_directory, f"{source}_{data_type}Data_stats.txt")



    # Save the tissue mask as a PNG file
    if source == "TCGA":
        output_file_name = patient_id + '.png'
    else:
        output_file_name = patient_id + '_' + slide_id + '.png'

    
    output_path_mask = os.path.join(mask_output_dir, output_file_name)
    img = Image.fromarray(inverted_tissue_mask)
    img.save(output_path_mask)

    # Save the RGB mask as a PNG file
    output_path_rgb_mask = os.path.join(rgb_mask_output_dir, output_file_name)
    img_rgb = Image.fromarray(rgb_mask, 'RGB')
    img_rgb.save(output_path_rgb_mask)


    # Save the eroded mask as a PNG file if the erosion kernel size and iterations are not zero
    if FLAGS.erosion_size != 0 and FLAGS.erosion_iterations != 0:
        eroded_mask = get_eroded_mask(output_path_mask, kernel_size=FLAGS.erosion_size, iterations=FLAGS.erosion_iterations, inverted=True)
        cv2.imwrite(os.path.join(eroded_mask_output_dir, output_file_name), eroded_mask)

    # Save the dilated mask as a PNG file if the dilation kernel size and iterations are not zero
    if FLAGS.dilation_size != 0 and FLAGS.dilation_iterations != 0:
        dilated_mask = get_dilated_mask(output_path_mask, kernel_size=FLAGS.erosion_size, iterations=FLAGS.erosion_iterations, inverted=True)
        cv2.imwrite(os.path.join(dilated_mask_output_dir, output_file_name), dilated_mask)
    # save thumbnail for the rgb mask
    thumbnail = Image.fromarray(slide_rgb)
    thumbnail.save(os.path.join(rgb_mask_output_dir, output_file_name.replace('.png', '_thumbnail.png')))

    print(f"Resolution for {patient_id}: {val_x}")
    print(f"Tissue area: {round(Mask_area_px * real_length * real_length / 1000000,3)} mm^2")
    print(f"Total area: {round(im_area_px * real_length * real_length / 1000000,3)} mm^2")
    print(f"Mask/Area percentage: %{round(Mask_area_ratio * 100,3)}")
    print(f"Data type: {data_type}")
    print("-----------------------------------")

    
    if txt_file_path_created_bool_list[source][data_types_list.index(data_type)] == False:
        with open(txt_file_path, 'w') as f:
            f.write("# path of mask, path of slide, Mask Image level(level_x = 2**x), Real Length in microns, Patient ID, Slide ID, Source Name, ROI(X-Y-Width-Height) \n")
        txt_file_path_created_bool_list[source][data_types_list.index(data_type)] = True
        
        if FLAGS.erosion_size != 0 and FLAGS.erosion_iterations != 0:
            with open(eroded_txt_file_path, 'w') as f:
                f.write("# path of mask, path of slide, Mask Image level(level_x = 2**x), Real Length in microns, Patient ID, Slide ID, Source Name, ROI(X-Y-Width-Height) \n")
        
        if FLAGS.dilation_size != 0 and FLAGS.dilation_iterations != 0:
            with open(dilated_txt_file_path, 'w') as f:
                f.write("# path of mask, path of slide, Mask Image level(level_x = 2**x), Real Length in microns, Patient ID, Slide ID, Source Name, ROI(X-Y-Width-Height) \n")
        
    if Data_stats_path_created_bool_dict[source][data_types_list.index(data_type)] == False:
        with open(Data_stats_path, 'w') as f:
            f.write("# Patient ID, Slide ID, Tissue/Total %, Tissue Area(mm), Bg/Total %, Bg Area(mm), Total Area(mm), Tissue-Size(Mb), Bg-Size(Mb), FileSize(Mb) \n")
        Data_stats_path_created_bool_dict[source][data_types_list.index(data_type)] = True
        
    # Write the pathname of the tissue mask image to the text file
    with open(txt_file_path, 'a') as f:
        f.write(output_path_mask + ',' + slide_path + ',' + str(normalized_level) + ',' + str(real_length) + ',' + patient_id + ',' + slide_id + ',' + source + ',' + roi_str + '\n')
    
    if FLAGS.erosion_size != 0 and FLAGS.erosion_iterations != 0:
        with open(eroded_txt_file_path, 'a') as f:
            f.write(os.path.join(eroded_mask_output_dir, output_file_name) + ',' + slide_path + ',' + str(normalized_level) + ',' + str(real_length) + ',' + patient_id + ',' + slide_id + ',' + source + ',' + roi_str + '\n')
    
    if FLAGS.dilation_size != 0 and FLAGS.dilation_iterations != 0:
        with open(dilated_txt_file_path, 'a') as f:
            f.write(os.path.join(dilated_mask_output_dir, output_file_name) + ',' + slide_path + ',' + str(normalized_level) + ',' + str(real_length) + ',' + patient_id + ',' + slide_id + ',' + source + ',' + roi_str + '\n')
        
    with open(Data_stats_path, 'a') as f:
        f.write(patient_id + ',' + slide_id + ',' + str(round(Mask_area_ratio * 100,2)) + ',' + str(round(Mask_area_px * real_length * real_length / 1000000,2)) + ',' + str(round((1 - Mask_area_ratio) * 100, 2)) + ',' + str(round((1 - Mask_area_ratio) * real_length * real_length / 1000000,2)) + ',' + str(round(im_area_px * real_length * real_length / 1000000,2)) + ',' + str(Mask_area_ratio * os.path.getsize(slide_path) // (2**20)) + ',' + str((1 - Mask_area_ratio) * os.path.getsize(slide_path) // (2**20)) + ',' + str(os.path.getsize(slide_path) // (2**20)) +'\n')
#path of mask, path of slide, Image level(level_x = 2**x), Real Lenght in microns,mask size, Patient ID, Slide ID