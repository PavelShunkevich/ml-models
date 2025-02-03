# CIFAR-10 Image Classification Model based on ResNet with Dropout

This model is trained to classify images from the CIFAR-10 dataset using a ResNet with Dropout architecture. It is capable of distinguishing between 10 different object classes.

**Training Date:** 2025-01-25

## Model Architecture

A custom convolutional network using:

* Initial conv, batch norm, and ReLU.
* Two residual blocks with stride 2.
* Adaptive average pooling.
* Fully connected layers with dropout and ReLU.

Total number of trainable parameters: 1 185 290

## Dataset

Dataset: CIFAR-10

CIFAR-10 consists of 60,000 32x32 color images divided into 10 classes.

The dataset is split into training (50,000 images) and testing (10,000 images) sets.

Data normalization was used.

## Training Process

*   Optimizer: Adam
*   Loss function: CrossEntropyLoss
*   Initial Learning rate: 0.01
*   Number of epochs: 100
*   Batch size: 256

## Performance Evaluation

*   Metric: Accuracy
*   Accuracy on the test set: 0.8111

## Model Usage

The model is can be using the following notebook: [test_model.ipynb](test_model.ipynb)
