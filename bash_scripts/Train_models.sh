
echo "Start training LeNet5 with TCGA_Level4_128 TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_4_size_128.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_4_size_128.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 128 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L4_128  --device_num 0 

echo "Start training LeNet5 with TCGA_Level4_64TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_4_size_64.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_4_size_64.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 64 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L4_64  --device_num 0 

echo "Start training LeNet5 with TCGA_Level5_64 TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_5_size_64.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_5_size_64.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 64 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L5_64  --device_num 0 

echo "Start training LeNet5 with TCGA_Level5_32TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_5_size_32.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_5_size_32.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 32 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L5_32  --device_num 0 

echo "Start training LeNet5 with TCGA_Level6_32TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_6_size_32.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_6_size_32.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 32 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L6_32  --device_num 0 

echo "Start training LeNet5 TCGA_Level6_16 TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/train/allpatches__level_6_size_16.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Eroded-Dilated-Outputs/Cropped_slides/TCGA/val/allpatches__level_6_size_16.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 16 --model_mode train --model_type LeNet5  --data_source TCGA --Level_patch L6_16  --device_num 0 

