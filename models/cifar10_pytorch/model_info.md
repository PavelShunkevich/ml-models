# Model info

# CIFAR-10 Image Classification Model based on ResNet-18

This model is trained to classify images from the CIFAR-10 dataset using a ResNet-18 architecture. It is capable of distinguishing between 10 different object classes.

**Training Date:** 2023-10-27

## Model Architecture

The model is based on the ResNet-18 architecture, which consists of a set of convolutional layers, subsampling layers, and fully connected layers. It includes ResNet blocks with skip connections.

*   Convolutional layers: `Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)`, ...
*   Max pooling layers: `MaxPool2d(kernel_size=2, stride=2)`
*   Fully connected layers: `Linear(in_features=512, out_features=10)`

Total number of trainable parameters: 11,689,512

## Dataset

Dataset: CIFAR-10

CIFAR-10 consists of 60,000 32x32 color images divided into 10 classes.

The dataset is split into training (50,000 images) and testing (10,000 images) sets.

Data normalization was used.

## Training Process

*   Optimizer: Adam
*   Loss function: CrossEntropyLoss
*   Initial Learning rate: 0.001
*   Number of epochs: 50
*   Batch size: 64

## Performance Evaluation

*   Metric: Accuracy
*   Accuracy on the test set: 0.88

## Model Usage

The model can be loaded using the following code:

```python
import torch
model = torch.load('cifar10_model.pth')
model.eval()