from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import threading

app = Flask(__name__)

# ==================== LOAD MODEL & PREPROCESSORS ====================
print("Loading model and preprocessors...")

model = tf.keras.models.load_model("models/gesture_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler: StandardScaler = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Model loaded successfully!")

# ==================== MEDIAPIPE SETUP ====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==================== GLOBAL CAMERA ====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("❌ ERROR: Webcam not accessible")

# ==================== SHARED STATE ====================
lock = threading.Lock()
current_prediction = "Waiting for hands..."

# ==================== LANDMARK NORMALIZATION ====================
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]  # wrist as origin
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()  # 63

def extract_two_hand_features(results):
    """
    Always returns 126 features:
    [63 left-hand | 63 right-hand]
    Missing hand → zero vector
    """
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            vec = normalize_landmarks(lm.landmark)
            if handedness.classification[0].label == "Left":
                left = vec
            else:
                right = vec

    return np.concatenate([left, right])  # 126

# ==================== VIDEO STREAM GENERATOR ====================
def generate_frames():
    global current_prediction

    with mp_hands.Hands(
        max_num_hands=2,   # ✅ DOUBLE HAND
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        print("✅ Streaming started → http://localhost:5000")

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label = "No hand detected"

            if results.multi_hand_landmarks:
                # Draw both hands
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS
                    )

                features = extract_two_hand_features(results)
                X = scaler.transform([features])
                preds = model.predict(X, verbose=0)

                idx = int(np.argmax(preds))
                label = label_encoder.inverse_transform([idx])[0]

                with lock:
                    current_prediction = label

                cv2.putText(
                    frame,
                    label,
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    5
                )
            else:
                with lock:
                    current_prediction = "No hand detected"

            ret, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() + b"\r\n"
            )

# ==================== ROUTES ====================
@app.route("/")
def index():
    gestures = sorted(label_encoder.classes_.tolist())
    return render_template("index.html", gestures=gestures)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/current_label")
def current_label_api():
    with lock:
        return jsonify({"label": current_prediction})

# ==================== RUN ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=False, threaded=True)
