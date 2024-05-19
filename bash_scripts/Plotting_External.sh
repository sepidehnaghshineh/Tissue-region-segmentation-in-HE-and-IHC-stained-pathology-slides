echo "Start plotting for HE_breast_tissues: "
python3 plotting_analysis_External.py --data_source TCGA  --ext_data_source External_for_TCGA    --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 

python3 plotting_analysis_External.py --data_source HEROHE --ext_data_source External_for_HEROHE    --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 

python3 plotting_analysis_External.py --data_source BAU_HE  --ext_data_source External_for_BAU_HE    --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 

python3 plotting_analysis_External.py --data_source HER2_HE  --ext_data_source External_for_HER2_HE    --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 


echo "Start plotting for IHC_breast_tissues: "

python3 plotting_analysis_External.py --data_source BAU_IHC  --ext_data_source HER2_IHC   --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 

python3 plotting_analysis_External.py --data_source HER2_IHC --ext_data_source BAU_IHC   --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 


echo "Start plotting for Combined_Level4_128 TL: "
python3 plotting_analysis_External.py --data_source Combined --ext_data_source External_for_Combined  --Level_patch L4_128  --model_type LeNet5 --train_type just_Student --num_epochs 1000 
