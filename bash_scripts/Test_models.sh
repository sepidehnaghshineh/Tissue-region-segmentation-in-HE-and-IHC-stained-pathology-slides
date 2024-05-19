"""" The code for testing the rtrained models. here we test the TCGA dataset with different patch sizes and levels as examples."""

echo "Start testing LeNet5 with TCGA_Level4_128: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_4_size_128_.txt --batch_size 1024 --input_size 128 --model_mode test --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L4_128 

echo "Start testing LeNet5 with TCGA_Level4_64: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_4_size_64_.txt --batch_size 1024 --input_size 64 --model_mode test --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L4_64 

echo "Start testing LeNet5 with TCGA_Level5_64: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_5_size_64_.txt --batch_size 1024 --input_size 64 --model_mode ttest --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L5_64 

echo "Start testing LeNet5 with TCGA_Level5_3: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_5_size_32_.txt --batch_size 1024 --input_size 32 --model_mode test --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L5_32 

echo "Start testing LeNet5 with TCGA_Level6_3: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_6_size_32_.txt --batch_size 1024 --input_size 32 --model_mode test --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L6_32 

echo "Start testing LeNet5 TCGA_Level6_16: "
python3 CNN-Test.py --dataset_dir ./Dataset_Prep/data_preparation_results/Cropped_slides/TCGA/test/allpatches__level_6_size_16_.txt --batch_size 1024 --input_size 16 --model_mode test --model_type LeNet5 --train_type just_Student --data_source TCGA --Level_patch L6_16 

