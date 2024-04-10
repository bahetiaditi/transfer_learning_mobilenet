Skin Lesion Segmentation Using MobileNet
Overview
This project aims to segment skin lesions in images using a MobileNet encoder pretrained on ImageNet and a custom decoder. It explores two main approaches: feature extraction and fine-tuning.

Dataset
Utilizes the ISIC 2016 dataset with 900 training images and 379 test images, including segmented masks. Images are preprocessed to a uniform size of 128x128 pixels.

project/
│
├── models/                  # Model definitions
│   ├── mobilenet.py         # MobileNet encoder
│   └── segnet.py            # SegNet model including custom decoder
│
├── utils/                   # Utility functions
│   ├── dataset.py           # Custom dataset loader
│   ├── metrics.py           # IoU and Dice score calculations
│   └── visualization.py     # Visualization tools
│
├── train_feature_extraction.py # Training script for feature extraction
├── train_fine_tuning.py     # Training script for fine-tuning
├── evaluate.py              # Evaluation script
└── README.md                # This README file

Experiments
Feature Extraction: Keeps the MobileNet encoder frozen and trains only the custom decoder.
Fine-Tuning: Unfreezes the MobileNet encoder for end-to-end training.

Results
Experiment 1 showed promising results with an average IoU of 0.7585 and Dice score of 0.8025 using BCEWithLogitsLoss.
Experiment 2 improved performance with fine-tuning, achieving an average IoU of 0.7659 and Dice score of 0.8094.

Colab Notebook
For a detailed walkthrough of the experiments and code, refer to the Colab Notebook.
