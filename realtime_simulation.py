import pandas as pd
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model
import os

WINDOW = 10   # must match training window

CSV_FILE = "traffic_data.csv"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
MODEL_FILE = "traffic_lstm_model.h5"

# -------------------------------
# Load data, scaler, model
# -------------------------------
df = pd.read_csv(CSV_FILE)
df = df[df["distance_cm"] > 0].reset_index(drop=True)

scaler = joblib.load(SCALER_FILE)
model = load_model(MODEL_FILE)

label_encoder = None
if os.path.exists(LABEL_ENCODER_FILE):
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

print("Starting REAL-TIME Simulation...\n")

# Scale distances using DataFrame to keep feature name
scaled_vals = scaler.transform(df[["distance_cm"]])  # shape (N,1)

# -------------------------------
# REAL-TIME STREAMING LOOP
# -------------------------------
buffer = []    # store last 10 scaled distances

for i in range(len(scaled_vals)):
    # Append new value to buffer
    buffer.append(scaled_vals[i])

    # Keep only last WINDOW readings
    if len(buffer) > WINDOW:
        buffer.pop(0)

    # Only predict when buffer full
    if len(buffer) == WINDOW:
        X = np.array(buffer).reshape(1, WINDOW, 1)

        # Predict
        pred = model.predict(X, verbose=0)[0]

        # Multi-class softmax case
        if pred.shape[-1] > 1:
            class_index = int(np.argmax(pred))
            if label_encoder:
                label = label_encoder.inverse_transform([class_index])[0]
            else:
                # assume 0=CLEAR, 1=MODERATE, 2=HEAVY
                mapping = {0: "CLEAR", 1: "MODERATE", 2: "HEAVY"}
                label = mapping.get(class_index, class_index)

            print(f"Distance={df.loc[i,'distance_cm']:.2f}  →  {label}   probs={pred}")

        else:
            # Sigmoid case (binary)
            prob = float(pred[0])
            label = "HEAVY" if prob >= 0.5 else "CLEAR"
            print(f"Distance={df.loc[i,'distance_cm']:.2f}  →  {label}   prob={prob:.3f}")

    # Delay to simulate live streaming
    time.sleep(0.25)
