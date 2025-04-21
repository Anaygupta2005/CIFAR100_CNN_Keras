# CIFAR-100 CNN Image Classifier

This project implements a **Convolutional Neural Network (CNN)** from scratch using Keras to classify images from the **CIFAR-100 dataset**. It demonstrates core deep learning principles such as image preprocessing, model design, evaluation, and error analysis.

The model is trained to distinguish between **100 fine-grained object classes**, ranging from animals and plants to everyday objects and vehicles.

## Dataset: CIFAR-100

- 60,000 color images of size **32x32 pixels**
- Split into 50,000 training and 10,000 test images
- 100 distinct object categories (each image belongs to one of them)

## Key Features

- Built a CNN architecture from the ground up using Keras
- Preprocessing includes normalization and one-hot encoding of labels
- Trained on all 100 classes with dropout regularization
- Plotted training/validation loss and accuracy across epochs
- Visualized both correct and failed predictions to interpret model performance

## Architecture Overview

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.35))

model.add(Conv2D(100, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(100, activation='softmax'))

