# collect_data.py  (DOUBLE HAND VERSION)
import cv2
import csv
import time
from pathlib import Path
import numpy as np

from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ==========================
# Paths
# ==========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "gestures.csv"

# ==========================
# Landmark Normalization (63 features per hand)
# ==========================
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Wrist as origin
    coords -= coords[0]

    # Scale
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist

    return coords.flatten()  # 63 values

# ==========================
# Main Program
# ==========================
def main():
    gesture_name = input("Enter gesture name (e.g. 'hello', 'thank_you'): ").strip()
    if not gesture_name:
        print("No name entered. Exiting.")
        return

    num_samples = int(input("How many samples to record? (e.g. 200): ") or 200)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # MediaPipe Hands – TWO HANDS
    hands_detector = Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    collected = 0
    print("Get ready... starting in 3 seconds.")
    time.sleep(3)

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        # Default: zero vectors
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                vec = normalize_landmarks(hand_landmarks.landmark)

                if label == "Left":
                    left_hand = vec
                else:
                    right_hand = vec

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, HAND_CONNECTIONS
                )

            # Combine → 126 features
            feature_vector = np.concatenate([left_hand, right_hand])

            write_header = not CSV_PATH.exists()
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    header = ["label"] + [f"f{i}" for i in range(126)]
                    writer.writerow(header)
                writer.writerow([gesture_name] + feature_vector.tolist())

            collected += 1

            cv2.putText(
                frame,
                f"Collected: {collected}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Show one or two hands clearly!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Collecting DOUBLE HAND data (Press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands_detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved {collected} samples for '{gesture_name}' to {CSV_PATH}")

if __name__ == "__main__":
    main()
