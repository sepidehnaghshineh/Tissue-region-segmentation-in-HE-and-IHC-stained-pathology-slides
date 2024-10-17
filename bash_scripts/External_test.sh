
echo "Start external validation of LeNet5_Combined_Level4_128: "
python3 CNN-External.py --dataset_dir ./External_Experiments/External_for_Combined/allpatches__level_4_size_128.txt --batch_size 1024 --input_size 128 --model_mode external_test --model_type LeNet5 --train_type just_Student --data_source Combined --ext_data_source External_for_Combined --Level_patch L4_128 
