---

# Face Mask Detection using MobileNet

---

This repository contains code for detecting face masks using a Convolutional Neural Network (CNN) and MobileNet for transfer learning. The project includes dataset creation, training the model, and evaluating the performance.

## Overview

The project consists of two main parts:
1. **Dataset Creation**: Collecting and preprocessing the dataset for training the model.
2. **Model Training and Evaluation**: Using MobileNet for transfer learning to create a face mask detection model.

## Contents

- `CNN Face Mask Detection (Dataset Creation).ipynb`: Notebook for dataset creation.
- `CNN_Face_Mask_Detection.ipynb`: Notebook for training and evaluating the model.

## Dataset

The dataset used for this project can be found on Kaggle: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

## Setup

To run the code in this repository, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Sankalpa0011/face-mask-detection.git
    cd face-mask-detection
    ```

2. Install the required libraries:
    ```bash
    pip install numpy matplotlib opencv-python keras tensorflow
    ```

3. Download the dataset from Kaggle and extract it to the `data` directory.

## Usage

### Dataset Creation

1. Open the `CNN Face Mask Detection (Dataset Creation).ipynb` notebook.
2. Run the cells to create and preprocess the dataset.

### Model Training and Evaluation

1. Open the `CNN_Face_Mask_Detection.ipynb` notebook.
2. Run the cells to train the model and evaluate its performance.

### Prediction

1. Load a new image and preprocess it using the `preprocess_image` function.
2. Use the trained model to predict if the person in the image is wearing a mask or not.

## Acknowledgements

- The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

---
