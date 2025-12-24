import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
import time
import os

# ============================================================
# THREADED SPEECH ENGINE (WORKING + AUTO CLEANUP)
# ============================================================
import threading
import queue
import asyncio
import edge_tts
from playsound import playsound
import tempfile

VOICE = "en-US-AriaNeural"
speech_queue = queue.Queue()

def audio_worker():
    """Background thread for TTS with guaranteed cleanup."""
    while True:
        text = speech_queue.get()
        if text is None:
            break

        # Create temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            file_path = tmp.name

        try:
            async def generate_audio():
                tts = edge_tts.Communicate(text, VOICE)
                await tts.save(file_path)

            asyncio.run(generate_audio())

            # Play audio (blocking, safe)
            playsound(file_path)

        finally:
            # Guaranteed cleanup
            try:
                os.remove(file_path)
            except:
                pass

        speech_queue.task_done()

# Start audio thread
audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

def speak(text):
    speech_queue.put(text)

# ============================================================
# LOAD MODEL & SCALER
# ============================================================
print("Loading gesture model & encoders...")

model = tf.keras.models.load_model("models/gesture_model.keras")

with open("models/scaler.pkl", "rb") as f:
    scaler: StandardScaler = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Loaded successfully!")
print("Realtime gesture recognition started. Press 'q' to quit.")

# ============================================================
# MEDIAPIPE HANDS
# ============================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]  # wrist as origin
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()  # 63

def extract_two_hand_features(results):
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

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    stable_label = None
    stable_count = 0
    STABLE_FRAMES = 15

    last_spoken_label = None
    last_spoken_time = 0
    SPEAK_GAP = 1.2  # seconds

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS
                    )

                features = extract_two_hand_features(results)
                X = scaler.transform([features])
                preds = model.predict(X, verbose=0)

                idx = int(np.argmax(preds))
                label = label_encoder.inverse_transform([idx])[0]

                cv2.putText(
                    frame,
                    label,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 255, 0),
                    4
                )

                # -------- STABLE SPEAK LOGIC --------
                if label == stable_label:
                    stable_count += 1
                else:
                    stable_label = label
                    stable_count = 1

                now = time.time()
                if (
                    stable_count >= STABLE_FRAMES and
                    (label != last_spoken_label or now - last_spoken_time > SPEAK_GAP)
                ):
                    print("Speaking:", label)
                    speak(label)
                    last_spoken_label = label
                    last_spoken_time = now
                    stable_count = 0
                # ----------------------------------

            cv2.imshow("Double-Hand Gesture Recognition + Voice", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
