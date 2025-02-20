# Plant-disease-detection
This project helps us to identify the disease associated with plant.
# Plant Disease Detection using [CNN, OpenCV, TensorFlow, Streamlit]

This repository contains the code for a *Plant Disease Detection* project. The project aims to identify and classify plant diseases from leaf images using deep learning techniques.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Streamlit Application](#streamlit-application)
7. [Model Architecture](#model-architecture)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)
13. [References](#references)

## Overview

This project focuses on developing a model to detect plant diseases from leaf images using convolutional neural networks (CNNs). The model is trained to classify different types of diseases affecting plants, helping farmers and researchers in early detection and prevention.

## Features

- *Real-time Plant Disease Detection*: The model can identify diseases in real-time using a webcam or uploaded images.
- *High Accuracy*: Achieved through the use of advanced deep learning techniques.
- *User-friendly Interface*: An intuitive interface for users to interact with the model.

## Dataset

The dataset used in this project consists of images of healthy and diseased plant leaves. The dataset structure includes:

- *Training Data*: Organized in directories based on disease classes.
- *Labels*: Stored in plant_disease_data, which maps class indices to labels.
- *Test Data*: Placed in the data/test directory for evaluation purposes.

The dataset can be expanded to include additional plant species and diseases.

## Installation

Follow these steps to set up and run the project on your local machine:

### Prerequisites

Make sure you have the following installed:

- *Python 3.10*
- *pip* (Python package installer)
- *TensorFlow* (for deep learning)
- *OpenCV* (for image processing)
- *Streamlit* (for the web interface)

### Steps

1. *Clone the Repository:*
   
   bash
   git clone https://github.com/gowtham611/plant_disease.git
   cd plant_disease
   

2. *Install Dependencies:*
   
   bash
   pip install -r requirements.txt
   

## Usage

To run the application:

bash
python app.py


## Streamlit Application

The Streamlit app provides the following features:

- *Homepage*: Introduction to the project and instructions for users.
- *Real-Time Detection*: Identifies plant diseases using a webcam.
- *Image Upload*: Users can upload images of plant leaves for disease classification.
- *Disease Information*: Provides details on detected diseases and possible treatments.

## Model Architecture

The model follows a CNN-based approach for classification:

- *Convolutional Neural Network (CNN)*: Used for image-based disease detection.
- *Transfer Learning*: Pre-trained models like VGG16, ResNet, etc., can be used for better accuracy (though this model was trained from scratch).
- *Pooling Layers*: Reduce spatial dimensions.
- *Fully Connected Layers*: Perform final classification.
- *Convolutional Layers*: Extract features from images.

The model is implemented in datacollection.ipynb.

## Training

The training process involves:

- *Data Augmentation*: Rotation, flipping, and scaling to improve generalization.
- *Optimization Techniques*: Adam optimizer for faster convergence.
- *Loss Function*: Categorical cross-entropy for multi-class classification.

## Evaluation

The model is evaluated using:

- *Accuracy*
- *Precision, Recall, and F1-score*
- *Confusion Matrix*

## Results

The model achieves an accuracy of *95%* on the test dataset.

Sample predictions include:

- Input Image: Leaf with visible spots.
- Predicted Label: *Bacterial Blight*
- Confidence Score: *92%*

## Contributing

If you would like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -am 'Add new feature').
5. Push to the branch (git push origin feature-branch).
6. Create a new Pull Request.

## License

This project currently does not have a license. If you intend to make your work publicly available and reusable, consider adding an open-source license:

- *MIT License* (permissive and widely used)
- *Apache License 2.0* (permissive with patent rights)
- *GPLv3* (requires derivative works to be open-sourced)

Refer to [Choose a License](https://choosealicense.com/) for guidance.

## References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. OpenCV Documentation: https://opencv.org/
3. Streamlit Documentation: https://www.streamlit.io/
