# Real-Time Bi-Directional Indian Sign Language Translator and Recognition System

This project implements a **real-time bi-directional Indian Sign Language (ISL) communication system** that enables seamless interaction between sign language users and non-sign language users. The system supports:

- **Text / Audio → ISL animated video translation**
- **ISL gestures → Text and Audio recognition**

By combining computer vision, machine learning, and web technologies, the system provides an accessible and efficient solution for inclusive communication.

---

## Project Modes

### 1. Text / Audio to ISL Animated Video
Converts user-provided text or spoken audio into Indian Sign Language using pre-generated 2D animated ISL videos. This module ensures uniform and consistent gesture representation instead of using raw human videos.

### 2. ISL Gesture to Text / Audio
Recognises Indian Sign Language gestures from a live webcam feed and converts them into readable text and optional speech output using an Artificial Neural Network (ANN).

---

## Technology Stack

- **Backend**: Python, Flask  
- **Computer Vision**: OpenCV, MediaPipe  
- **Machine Learning**: TensorFlow / Keras (ANN), Scikit-learn  
- **Frontend**: HTML, CSS, JavaScript  
- **Audio Processing**: Browser Speech Recognition & Text-to-Speech  

---

## Prerequisites

- Python **3.10**
- Webcam (for gesture recognition)
- Google Chrome / Edge (recommended)

---

## Step 1: Install Dependencies

```bash
pip install flask opencv-python mediapipe tensorflow scikit-learn numpy pandas
```

### Running Mode 1: Text / Audio → ISL Animated Video
Terminal 1
```bash
cd Text_Audio_to_ISL
python app.py
Server URL:http://localhost:5000
```

### Running Mode 2: ISL Gesture → Text / Audio
Terminal 2
```bash
cd ISL_to_Text_Audio
python main_app.py
Server URL:http://localhost:5000
```
