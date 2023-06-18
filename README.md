# Dog Images Classification using Transfer Learning
This notebook builds a end-to-end multiclass image classifier using TensorFlow 2.0 and TensorFlow Hub.
![golden](https://github.com/KelvinJulius/dog_vision/assets/108222785/25954cba-fdd6-4f1a-9742-4094d549e9fe)

## Problem
Given a  dog picture. Guess what kind of dog it is?

## About the Data
Get the data for dog vision from Kaggle. Here is the link:
https://www.kaggle.com/c/dog-breed-identification/overview

## Evaluation
The evaluation metrics is probabilities output (multiclass output) from softmax activation.
https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation

## Features
* We're using deep learning/transfer learning because it's probably best way for images (unstructured data).
* There are 120 unique kind of dogs in this data.
* There are 10,000+ images in the training set (Images with labels)
* There are around 10,000+ images in the test set (Images have no labels)
  
## Result
Succesfully to handle overfitting using data augmentation. Get 84% and 82% for loss training score and validating score.

## Workflow of this project
* Preparing the data (Loading from Kaggle, preprocessing the images, and split the data into 3 sets (training, validating, testing))
* Choosing a model and Fitting the model (Using Tensorflow Hub and prevent the overfitting using `earlystopping` callback)
* Monitoring the performance of loss training and loss validating (using tensorboard callback)
* Improve the model (Using Image Augmentation for reducing the overfitting model)
* Save the model
