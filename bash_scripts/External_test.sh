echo "Start external validation of LeNet5_BAU_HE_Level4_128 : "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_BAU_HE/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test  --model_type LeNet5 --train_type just_Student --data_source BAU_HE  --ext_data_source External_for_BAU_HE --Level_patch L4_128 

echo "Start external validation of LeNet5_HER2_HE_Level4_128 : "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_HER2_HE/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source HER2_HE --ext_data_source External_for_HER2_HE --Level_patch L4_128 


echo "Start external validation of LeNet5_HEROHE_Level4_128 : "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_HEROHE/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source HEROHE --ext_data_source External_for_HEROHE --Level_patch L4_128 


echo "Start external validation of LeNet5_TCGA_Level4_128 : "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_TCGA/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source TCGA --ext_data_source External_for_TCGA --Level_patch L4_128 



echo "Start external validation of LeNet5_BAU_IHC_Level4_128 : "
python3 CNN-External.py --dataset_dir ./data_preparation_results/Cropped_slides/HER2_IHC/test/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source BAU_IHC --ext_data_source HER2_IHC --Level_patch L4_128 

echo "Start external validation of LeNet5_HER2_IHC_Level4_128 : "
python3 CNN-External.py --dataset_dir ./data_preparation_results/Cropped_slides/BAU_IHC/test/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source HER2_IHC --ext_data_source BAU_IHC --Level_patch L4_128 



echo "Start external validation of LeNet5_CAMELYON17_Level4_128 : "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_Combined/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source Combined --ext_data_source External_for_Combined --Level_patch L4_128 

echo "Start external validation of LeNet5_Combined_Level4_128: "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_Combined/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source Combined --ext_data_source External_for_Combined --Level_patch L4_128 
