# 🐱🐶 Cat vs Dog Image Classification

## 📌 Project Overview

This project is a deep learning-based image classification system that identifies whether an uploaded image contains a **Cat** or a **Dog**.

A Convolutional Neural Network (CNN) and transfer learning approach are used for image classification. The trained model is integrated with a Streamlit web application that allows users to upload an image and receive a prediction with a confidence score.

---

## 🎯 Objectives

- Classify images into Cat or Dog classes.
- Perform image preprocessing and normalization.
- Train a deep learning image classification model.
- Evaluate the model using validation accuracy and loss.
- Provide predictions for new images.
- Develop an easy-to-use Streamlit web interface.
- Make the project reproducible for other users.

---

## 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Matplotlib
- Streamlit

---

## 📂 Project Structure

```text
cat-dog-classification/
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
│
├── cat_dog_model.keras
├── cat_dog_best.keras
│
└── test_set/
    └── test_set/
        ├── cats/
        ├── dogs/
        └── cat_dog.csv