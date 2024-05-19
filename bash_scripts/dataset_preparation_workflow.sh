python3 ./Dataset_Prep/Tissue_masking_using_xml.py --input_dir  /media/snaghshineh/MAIL4TB_1/snaghshineh_data/data_snaghshineh/Camelyon17  

echo "WSI Cropping":


python3 ./Dataset_Prep/crop_all_leves_at_once/cropping.py --input_dir ./Dataset_Prep/data_preparation_results/Tissue_Masks/source_name_masks/train_tissue_mask_info.txt

python3 ./Dataset_Prep/crop_all_leves_at_once/cropping.py --input_dir ./Dataset_Prep/data_preparation_results/Tissue_Masks/source_name_masks/val_tissue_mask_info.txt

python3 ./Dataset_Prep/crop_all_leves_at_once/cropping.py --input_dir ./Dataset_Prep/data_preparation_results/Tissue_Masks/source_name_masks/test_tissue_mask_info.txt

