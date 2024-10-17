This repository is the official implementation of "TISSUE REGION SEGMENTATION IN H&E-STAINED AND IHC-STAINED PATHOLOGY SLIDES OF SPECIMENS FROM DIFFERENT ORIGINS".

We have developed a deep learning-based CNN model segmenting tissue regions in whole slide of a sample from its H&E stained, and IHC stained digital histopathology slides from different origins. the used CNN model is light-weighted with 19.8 Mb FLOPs and is proper for low cost implimantations and it
takes approximately 22 seconds to segment out a digital pathology slide.

We have successfully tested our CNN models on seven public and private different cohorts in The Cancer Genome Atlas (TCGA), HER2 grading challenge, HEROHE challenge, CAMELYON 17 challenge, PANDA challenge, a local Singapore cohort, and a local Turkey cohort. For all cohorts, we use the same model architecture. 
