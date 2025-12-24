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
Recognizes Indian Sign Language gestures from a live webcam feed and converts them into readable text and optional speech output using an Artificial Neural Network (ANN).

---

## Technology Stack

- **Backend**: Python, Flask  
- **Computer Vision**: OpenCV, MediaPipe  
- **Machine Learning**: TensorFlow / Keras (ANN), Scikit-learn  
- **Frontend**: HTML, CSS, JavaScript  
- **Audio Processing**: Browser Speech Recognition & Text-to-Speech  

---


## How to Run the Project

⚠️ **Both modules run independently and must be started in separate terminals.**

---

### Step 1: Install Dependencies

Ensure Python 3.10 is installed.

```bash
pip install flask opencv-python mediapipe tensorflow scikit-learn numpy pandas
Running Mode 1: Text / Audio → ISL Animated Video
Terminal 1
bash
Copy code
cd Text_Audio_to_ISL
python app.py
The server starts at:
http://localhost:5000

Allows:

Text input → ISL video

Audio input → ISL video

Speed control and video download

Running Mode 2: ISL Gesture → Text / Audio
Terminal 2
bash
Copy code
cd ISL_to_Text_Audio
python main_app.py
The server starts at:
http://localhost:9999

Allows:

Live webcam gesture recognition

Text output of detected gestures

Speech synthesis of recognized gestures

Training the Gesture Recognition Model (Optional)
If you want to retrain the ANN model:

bash
Copy code
cd ISL_to_Text_Audio
python train_model.py
Uses gestures.csv dataset

Saves trained model and preprocessors in models/

System Highlights
Real-time performance with low latency

Two-hand gesture support

ANN-based gesture classification (95% training accuracy)

Uniform 2D ISL animation output

Web-based and platform-independent

Applications
Assistive communication for deaf and hard-of-hearing users

Educational tools for ISL learning

Public service accessibility platforms

Human–computer interaction systems

Future Scope
Continuous sentence-level ISL recognition

Larger and more diverse ISL datasets

Facial expression and emotion analysis

Mobile and edge-device deployment

Multilingual translation support