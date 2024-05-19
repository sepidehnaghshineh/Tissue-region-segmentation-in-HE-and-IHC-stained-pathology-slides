import os
import sys
import numpy as np
import torch
import cv2
import time
from PIL import Image
import argparse
from openslide import OpenSlide
sys.path.append(os.getcwd())
from tools.LeNet5_different_inputSizes import select_model
from tools.PredictingTools import LoadData, TestToDataframe, SaveMask, heatmap
from tools.CroppingTools import read_slide_to_level, mask_slide_to_level, OTSU_slide_to_level
from tools.AnalyzingTools import Create_Overlay, mask_to_xml
from tools.ScoringTools import Jaccard_Index, Dice_Coefficient
parser = argparse.ArgumentParser(description='Predicting')

########################################################################################################

parser.add_argument('--input_dir', type=str, default=r"./Segmentation/segmentation_results/wsi.txt", help='a txt file containing path of the WSIs such as .svs, .mirx, .tiff, .ndpi files')
parser.add_argument('--out_dir', type=str, default=r"./Segmentation/segmentation_results", help='output directory')
parser.add_argument('--data_source', default='source_name', help='the source that you have got the WSIs', dest='data_source')
parser.add_argument('--voting', type=str, default=None, help='\"hard\", \"soft\" or None')
parser.add_argument('--resolution', type=int, default=4, help='resolution of the prediction, 1 is the cropsize, 2 is half of the cropsize, 3 is quarter of the cropsize etc.')

########################################################################################################


FLAGS = parser.parse_args()

mdl_basename = "LeNet5"
level_size:str = "L4_128"
data_source:str = FLAGS.data_source

mask_out_dir = FLAGS.out_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

txt_path = FLAGS.input_dir
slidepaths = []
xmlpaths = []
slidepaths = []

with open(txt_path) as f:
    for line in f:
        slidepath = line.strip()  # Remove leading/trailing spaces and newline characters
        xmlpath = slidepath.rsplit('.', 1)[0] + '.xml'
        
        xmlpaths.append(xmlpath)
        slidepaths.append(slidepath)
        
pred_resolution = FLAGS.resolution #it is for changing the size of stride, bigger number means smaller stride.
# 1 is the cropsize, 2 is half of the cropsize, 3 is quarter of the cropsize etc.



voting = FLAGS.voting

model_dir = os.path.join(mask_out_dir, mdl_basename)

mask_out_dir = os.path.join(model_dir, f"resolution_{pred_resolution}")
mask_out_dir = os.path.join(mask_out_dir, data_source)
os.makedirs(mask_out_dir, exist_ok=True)

scores_dir = os.path.join(mask_out_dir, 'scores.txt')
with open(scores_dir, 'w') as fil:
    fil.write("slide_name, jaccard_score, dice_coef\n")

otsu_scores_dir = os.path.join(mask_out_dir, 'OTSU_scores.txt')
with open(scores_dir, 'w') as fil:
    fil.write("slide_name, jaccard_score, dice_coef\n")


# './results/{FLAGS.data_source}/{FLAGS.Level_patch}/trained_models/{FLAGS.model_type}.pth'

model_level, model_cropsize = level_size[1:].split("_")
model_level, model_cropsize = int(model_level), int(model_cropsize)
model_path = r"Segmentation/model/model1/LeNet5Segmentation.pth"
means = np.load(r"Segmentation/model/model1/means.npy")
stds = np.load(r"Segmentation/model/model1/stds.npy")

stride = int(model_cropsize//2**(pred_resolution-1))

wanted_rlength = 2**(model_level-2) #we do -2 because our model base mpp is 0.25
#print("wanted_rlength: ", wanted_rlength)

model = select_model(model_cropsize)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

with open(scores_dir, 'a') as scores_file:
    with open(otsu_scores_dir, 'a') as otsu_scores_file:
        for slide_path, xml_path in zip(slidepaths, xmlpaths):
            start_time = time.time()
            slide_name = os.path.basename(slide_path).split('.')[0]
            print(slide_name)
            save_dir = os.path.join(mask_out_dir, slide_name)
            os.makedirs(save_dir, exist_ok=True)
            predicted_mask_path = os.path.join(save_dir,'mask.png')
            heatmap_mask_path = os.path.join(save_dir,'heatmap.png')
            heatmap_mask_legends_path = os.path.join(save_dir,'heatmap_leg.png')
            
            slide = OpenSlide(slide_path)
            try:
                current_res = float(slide.properties.get('openslide.mpp-x'))
            except:
                try:
                    res_type = slide.properties.get("tiff.ResolutionUnit")
                    if res_type == "centimeter":
                        numerator = 10000
                    elif res_type == "inch":
                        numerator = 25400
                    current_res = numerator / float(slide.properties.get("tiff.XResolution"))
                except:
                    raise Exception('Unknown Val_x')
            
            if current_res < 0.3:  # resolution:0.25um/pixel
                current_res = 0.25
            elif current_res < 0.6:  # resolution:0.5um/pixel
                current_res = 0.5
                
            xml_downscale = wanted_rlength / current_res
            img, downscale = read_slide_to_level(slide, rlenght=wanted_rlength)
            img_height, img_width = img.height, img.width
            img.save(os.path.join(save_dir,'original.png'))
            img = np.array(img)
            
            dataloader = LoadData(img_arr=img, cropsize=model_cropsize, stride=stride, means=means, stds=stds)

            pred_df = TestToDataframe(model, device, dataloader)
            pred_df.to_csv(os.path.join(save_dir,'preds.csv'), index=False)
            SaveMask(pred_df, img_height, img_width, model_cropsize, predicted_mask_path, voting=voting, class_weights=(0.75, 0.25))
            heatmap(pred_df, img_height, img_width, model_cropsize, heatmap_mask_legends_path)
            inverted_mask_im = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
            mask_im = cv2.bitwise_not(inverted_mask_im)
            
            overlay_im = Create_Overlay(Image.fromarray(img), Image.fromarray(mask_im), fill=(100, 100, 250, 70))
            overlay_im.save(os.path.join(save_dir,'overlay.png'))
            
            mask_to_xml(mask_im, os.path.join(save_dir, slide_name + '.xml'), downscale_factor=downscale)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Time taken for {slide_name}: {time_taken} seconds") 
            
            truth_mask = mask_slide_to_level(xml_path, (img_height, img_width), xml_downscale)
            OTSU_mask = OTSU_slide_to_level(slide, (img_height, img_width), wanted_rlenght=wanted_rlength)
            inverted_OTSU_mask = cv2.bitwise_not(OTSU_mask)
            # Save the inverted OTSU mask
            cv2.imwrite(os.path.join(save_dir, 'inverted_OTSU_mask.png'), inverted_OTSU_mask)  # Save inverted OTSU mask here
            inverted_truth_mask = cv2.bitwise_not(truth_mask)
            cv2.imwrite(os.path.join(save_dir, 'truth_mask.png'), inverted_truth_mask)
            # turn the image to grayscale if it is in rgb
            truth_mask = np.array(truth_mask)
            mask_im = np.array(mask_im)
            
            OTSU_overlay = Create_Overlay(Image.fromarray(img), Image.fromarray(OTSU_mask), fill=(100, 220, 100, 140))
            OTSU_overlay.save(os.path.join(save_dir,'OTSU_overlay.png'))
            print(truth_mask.shape)
            print(OTSU_mask.shape)
        
            
            otsu_jaccard_score = Jaccard_Index(truth_mask, OTSU_mask)
            otsu_dice_coef = Dice_Coefficient(OTSU_mask, truth_mask)
            
            print("mask_im.shape: ", mask_im.shape)
            jaccard_score = Jaccard_Index(truth_mask, mask_im)
            dice_coef = Dice_Coefficient(mask_im, truth_mask)
            print("jaccard_score: ", jaccard_score)
            print("dice_coef: ", dice_coef)
            scores_file.write(slide_name + ',' + str(jaccard_score) + ',' + str(dice_coef) + '\n')
            otsu_scores_file.write(slide_name + ',' + str(otsu_jaccard_score) + ',' + str(otsu_dice_coef) + '\n')
            slide.close()
            print("done with: " + slide_name)
            print("------------------------------------------------------")
        