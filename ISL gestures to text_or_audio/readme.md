# ISL Gesture to Text/Audio Recognition

This module implements a real-time Indian Sign Language (ISL) gesture recognition system that converts hand gestures into readable text and optional speech output. The system is designed to improve communication between the deaf or hard-of-hearing community and non-sign language users through an accessible web-based interface.

---

## Overview

The application captures live video input from a webcam and processes Indian Sign Language gestures using computer vision and machine learning techniques. MediaPipe Hands is used to extract hand landmark coordinates, which are normalized and classified using a trained Artificial Neural Network (ANN). The recognized gesture is displayed as text and can also be converted into speech for audio output.

---

## Key Features

- Real-time ISL gesture recognition using a webcam  
- Support for both single-hand and double-hand gestures  
- Hand landmark extraction using MediaPipe Hands  
- Artificial Neural Network (ANN) for gesture classification  
- Live text output of recognized gestures  
- Optional text-to-speech audio output  
- Web-based interface for ease of use  

---

## Technology Stack

- Python (3.10)  
- Flask  
- MediaPipe  
- OpenCV  
- TensorFlow / Keras  
- Scikit-learn  
- HTML, CSS, JavaScript  

---

## System Workflow

1. Capture live video frames from the webcam  
2. Detect and track hand landmarks using MediaPipe  
3. Normalize landmark coordinates for scale and position invariance  
4. Combine left and right hand features into a fixed-length feature vector  
5. Apply feature scaling using a pre-trained scaler  
6. Classify gestures using a trained ANN model  
7. Display the recognized gesture as text  
8. Convert the text output into speech (optional)  

---

## Directory Structure

ISL_to_Text_Audio/
├── data/
│ └── gestures.csv
├── models/
│ ├── gesture_model.keras
│ ├── scaler.pkl
│ └── labels.pkl
├── templates/
│ └── index.html
├── main_app.py
├── train_model.py
└── readme.md

yaml
Copy code

---

## Model Training

Gesture samples are collected using a webcam and stored in a CSV file containing normalized hand landmark features. An Artificial Neural Network (ANN) with multiple dense layers is trained using these features. Feature scaling is performed using `StandardScaler`, and gesture labels are encoded using `LabelEncoder`. The trained model achieves high accuracy and supports real-time inference.

---

## Application Usage

- Start the Flask server by running `main_app.py`  
- Open the application in a web browser  
- Perform ISL gestures in front of the webcam  
- View the recognized gesture as text on the interface  
- Listen to the gesture output using speech synthesis (if enabled)  

---

## Applications

- Assistive communication systems for the deaf and hard-of-hearing  
- Educational tools for learning Indian Sign Language  
- Human–computer interaction systems  
- Accessibility solutions for public and digital services  

---
