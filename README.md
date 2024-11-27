[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14131968.svg)](https://doi.org/10.5281/zenodo.14131968)

This repository is the official implementation of "TISSUE REGION SEGMENTATION IN H&E-STAINED AND IHC-STAINED PATHOLOGY SLIDES OF SPECIMENS FROM DIFFERENT ORIGINS".

We have developed a deep learning-based CNN model segmenting tissue regions in whole slide of a sample from its H&E stained, and IHC stained digital histopathology slides from different origins. the used CNN model is light-weighted with 19.8 Mb FLOPs and is proper for low cost implimantations and it
takes approximately 22 seconds to segment out a digital pathology slide.

We have successfully tested our CNN models on seven public and private different cohorts in The Cancer Genome Atlas (TCGA), HER2 grading challenge, HEROHE challenge, CAMELYON 17 challenge, PANDA challenge, a local Singapore cohort, and a local Turkey cohort from Bahcesehir Medical School. For all cohorts, we use the same model architecture. 

---

## Folder Structure

The repository is organized as follows:

```
├── Dataset_Prep/                  # Scripts and tools for preparing datasets
├── Models_Train_Test/             # Training and testing scripts for CNN models
│   ├── CNN_Train_Test/            # Subfolder for CNN-specific scripts
├── Segmentation/                  # Scripts and tools for tissue segmentation
├── Trained_Model_Analysis/        # Analysis and visualization of trained model results
├── bash_scripts/                  # Bash scripts for automated tasks
├── tools/                         # Additional utilities for the project
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies and requirements
```

### Folder Descriptions

1. **Dataset_Prep**:  
   Contains preprocessing scripts for preparing datasets, including data augmentation and normalization.

2. **Models_Train_Test**:  
   Contains training and testing scripts for building and evaluating CNN models.

3. **Segmentation**:  
   Includes inference scripts and tools to generate segmentation masks for WSIs.

4. **Trained_Model_Analysis**:  
   Provides tools to evaluate the performance of trained models, including metrics like Jaccard Index and Dice Coefficient.

5. **bash_scripts**:  
   Scripts for automating repetitive tasks such as running multiple experiments or generating reports.

6. **tools**:  
   Utility scripts and helper functions for various project tasks.

---
We will explain the following steps one-by-one:

# Tissue Segmentation Tool using LeNet5

This repository provides a tool for tissue segmentation in whole slide images (WSIs) using a deep learning model based on LeNet5. The tool processes WSIs, generates segmentation masks, and evaluates results using metrics like Jaccard Index and Dice Coefficient.

---

## Table of Contents
1. [Turkey Cohort](#turkey_cohort)  
1. [Features](#features)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Input Requirements](#input-requirements)  
5. [Output](#output)  
6. [Example Command](#example-command)  
7. [Citations](#citations)

---
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14131968.svg)](https://doi.org/10.5281/zenodo.14131968)

## Turkey Cohort
Digitized haematoxylin and eosin (H&E)-stained whole-slide-images (WSIs) of 72 Breast tissues and Digitized Immunehistochemistery (IHC)-stained whole-slide-images (WSIs) of 163 Breast tissues which were collected from patients at Bahcesehir Medical School in Turkey. H&E-stained and IHC-stained slides were scanned at 40× magnification (specimen-level pixel size 0.25μm × 0.25μm).

Slides were manually annotated and classified into two classes, tissue, and background, using the [the ASAP annotation tool](https://computationalpathologygroup.github.io/ASAP/).


## Features

- Processes WSIs in formats like `.svs`, `.mrxs`, `.tiff`, and `.ndpi`.  
- Generates segmentation masks and evaluation scores.  
- Supports "hard" and "soft" voting mechanisms for mask refinement.  
- Saves results and intermediate outputs for review.

---

## Installation

### Prerequisites

Ensure the following dependencies are installed:

- Python >= 3.8  
- PyTorch  
- OpenCV (`cv2`)  
- NumPy  
- PIL (Pillow)  
- OpenSlide-Python  
- argparse  

You can install dependencies with the following command:

```bash
pip install torch opencv-python-headless numpy pillow openslide-python argparse
```

### Cloning the Repository

Clone the repository to your local system:

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

---

## Usage

### 1. Prepare Input

Create a `.txt` file containing the paths of your WSIs (one file path per line). Ensure that the WSI files are accessible and supported formats like `.svs`, `.mrxs`, `.tiff`, or `.ndpi`.

### 2. Run the Script

Use the following command to execute the script:

```bash
python <script_name>.py --input_dir <path_to_txt> --out_dir <output_directory> --resolution <resolution> --voting <voting_type> --data_source <source_name>
```

Replace placeholders with appropriate values. See [Input Requirements](#input-requirements) for details.

---

## Input Requirements

- **Input Directory (`--input_dir`)**: Path to the `.txt` file listing the WSI paths.
- **Output Directory (`--out_dir`)**: Path to the folder where results will be saved.
- **Resolution (`--resolution`)**: Resolution scale for predictions (e.g., 1 for full crop size, 2 for half, etc.).
- **Voting (`--voting`)**: Optional voting mechanism (`hard`, `soft`, or `None`).
- **Data Source (`--data_source`)**: Descriptive label for the WSI dataset (e.g., "tumor_samples").

---

## Output

- **Segmentation Mask**: A binary mask saved as `mask.png` in the output directory.
- **Metrics**: Jaccard Index and Dice Coefficient scores saved in `scores.txt`.
- **Overlay and XML**: Optional outputs include overlay images and annotations in XML format.

---

## Example Command

Below is an example to run the script:

```bash
python predict.py --input_dir ./Segmentation/segmentation_results/wsi.txt --out_dir ./Segmentation/segmentation_results --resolution 4 --voting hard --data_source my_dataset
```

---

## Citations

If you use this code in your research, please cite the following sources:


---
