# Skin Lesion Segmentation Using MobileNet

## Overview
This project segments skin lesions in images using a pretrained MobileNet as an encoder and a custom decoder, exploring both feature extraction and fine-tuning approaches.

## Dataset
ISIC 2016 dataset with preprocessed images (128x128 pixels) including 900 training and 379 test images along with their segmented masks.

## Project Structure

project/
│
├── models/ # Contains model definitions
│ ├── mobilenet.py # MobileNet encoder setup
│ └── segnet.py # SegNet model including the custom decoder
│
├── utils/ # Utility functions for the project
│ ├── dataset.py # Custom dataset loader and transformations
│ ├── metrics.py # Calculation of IoU and Dice scores
│ └── visualization.py # Visualization utilities for results
│
├── train_feature_extraction.py # Script for feature extraction experiment
├── train_fine_tuning.py # Script for fine-tuning experiment
├── evaluate.py # Evaluation script for the model
├── notebook.ipynb #Colab notebook
└── README.md # Project documentation

## Experiments

1. **Feature Extraction:** Freezes the encoder and trains the decoder.
2. **Fine-Tuning:** Unfreezes the entire model for end-to-end training.

## Results Summary

- **Feature Extraction:** Achieved an average IoU of 0.7585 and Dice score of 0.8025.
- **Fine-Tuning:** Improved performance with an average IoU of 0.7659 and Dice score of 0.8094.

## Colab Notebook

For a comprehensive walkthrough, refer to the [Colab Notebook](https://colab.research.google.com/drive/1stLYaJcLd6rMSpQer2L2gYzvDVHAVZnH?usp=sharing)
