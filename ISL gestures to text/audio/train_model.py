# train_model.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

DATA_PATH = Path("data/gestures.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "gesture_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABELS_PATH = MODELS_DIR / "labels.pkl"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run collect_data.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"]).values.astype("float32")
    y = df["label"].values
    return X, y


def build_model(input_dim, num_classes):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading dataset...")
    X, y = load_data()

    print("Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    print("Building model...")
    model = build_model(X_train.shape[1], len(le.classes_))
    model.summary()

    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        verbose=2,
    )

    print("Saving model & encoders...")
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Labels saved to: {LABELS_PATH}\n")
    print("Training complete!")


if __name__ == "__main__":
    main()
