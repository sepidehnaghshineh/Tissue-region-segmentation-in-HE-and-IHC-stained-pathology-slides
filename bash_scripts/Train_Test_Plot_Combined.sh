
echo "Start training LeNet5 with Combined_data_Level4_128 TL: "
python3 CNN-Train.py --train_dataset_dir ./DATA_PREP/data_preparation_results/Cropped_slides/combined_data/train/allpatches__level_4_size_128.txt --val_dataset_dir ./DATA_PREP/data_preparation_results/Cropped_slides/combined_data/val/allpatches__level_4_size_128.txt --num_epochs 1000 --batch_size 1024 --learning_rate 0.0001 --input_size 128 --model_mode train --model_type LeNet5  --data_source Combined_data --Level_patch L4_128  --device_num 0 

echo "Start testing LeNet5 with Combined_data_Level4_128 TL: "
python3 CNN-Test.py --dataset_dir ./DATA_PREP/data_preparation_results/Cropped_slides/combined_data/test/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode test --model_type LeNet5  --data_source Combined_data --Level_patch L4_128   


echo "Start plotting for Combined_data_Level4_128 TL: "
python3 plotting_analysis.py --data_source Combined_data  --Level_patch L4_128     --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 




