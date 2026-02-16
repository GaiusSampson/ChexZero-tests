# CheXzero-tests


## Description

### overview

This project aims to test different implementations of models against the MMIC-CXR dataset to compare the performance, mainly CLIP based architecture using ViT/B-32, ResNet50, SwinTransformer and BioMedicalBERT.


### how to use
Download the project files and and install the dependancies by running "python -m pip install -r requirements.txt". To start run the run_preprocess file and make sure you add the paths to where you downloaded the dataset. 

To run the standard CLIP text enconder with ViT or RN edit the model name in train.py at line 149 to either "ViT-B/32" or "RN50". In train.py make sure CLIP is imported from "model" and not "swin_model". To evaluate these models move all the checkpoint files to the checkpoints directory and run zero_shot_evap.py, make sure the model name on line 88 of zero_shot.py is the one you used

For the Swin transformer make sure CLIP is imported from "swin_model" in train.py and run the run_swint.py file, to evaluate run the zero_shot_eval_swint.py file.

To run Swwin with biobert run the run the run_swint.py file with the flag --use_biobert

### disclaimer

Not all of the code in this project is written by me it is an adaptation of the original chexzero implementation from https://github.com/rajpurkarlab/CheXzero with some help from my acedemic supervisor

## Getting Started

### Dependencies

* you will need the MMIC-CXR dataset
* make sure the requirements.txt is installed
* this program was made on windows 11
