"""" The code for plotting the results of the experiments. here we plot the TCGA dataset with different patch sizes and levels as examples."""

echo "Start plotting for TCGA_Level4_128: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L4_128 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 

echo "Start plotting for TCGA_Level4_64: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L4_64 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 


echo "Start plotting for TCGA_Level5_64: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L5_64 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 


echo "Start plotting for TCGA_Level5_32: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L5_32 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 


echo "Start plotting for TCGA_Level6_32: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L6_32 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 


echo "Start plotting for TCGA_Level6_16: "
python3 plotting_analysis.py --data_source TCGA --Level_patch L6_16 --model_type LeNet5  --num_epochs 1000 --ext_data_source _ 


