# predict_from_csv.py  -- replace your current file with this
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

WINDOW = 10   # must match preprocess_traffic_data.py

# Files (assumed in same folder)
CSV = "traffic_data.csv"
SCALER = "scaler.pkl"
LABEL_ENCODER = "label_encoder.pkl"
MODEL = "traffic_lstm_model.h5"

# Load CSV
df = pd.read_csv(CSV)
if "distance_cm" not in df.columns:
    raise SystemExit("CSV must contain column 'distance_cm'")

# Load scaler (joblib)
if os.path.exists(SCALER):
    scaler = joblib.load(SCALER)
else:
    raise SystemExit(f"Missing scaler file: {SCALER}")

# Load label encoder if exists
le = None
if os.path.exists(LABEL_ENCODER):
    le = joblib.load(LABEL_ENCODER)

# Load model
model = load_model(MODEL)

print("Data rows:", len(df))
# Remove invalid rows (if any) like negative distances (optional)
df = df[df["distance_cm"] > 0].reset_index(drop=True)

# Scale entire series using DataFrame so scaler keeps feature names
scaled = scaler.transform(df[["distance_cm"]])   # shape (N,1)

# Build sliding windows
X = []
indexes = []   # index in original df corresponding to the window's 'target' time (the sample after the window)
for i in range(len(scaled) - WINDOW):
    seq = scaled[i:i+WINDOW]          # shape (WINDOW, 1)
    X.append(seq)
    indexes.append(i + WINDOW)        # predict for time index i+WINDOW

if len(X) == 0:
    raise SystemExit("Not enough rows to form one window. Need at least WINDOW+1 rows.")

X = np.array(X)         # (n_samples, WINDOW, 1)
print("Built windows:", X.shape)

# Predict
preds = model.predict(X, verbose=0)   # shape (n_samples, n_classes) or (n_samples,1)
for i, p in enumerate(preds):
    idx = indexes[i]
    last_dist = float(df.loc[idx, "distance_cm"])
    # if model was softmax with 3 classes
    if p.ndim == 1 or (isinstance(p, np.ndarray) and p.shape[-1] > 1):
        label_idx = int(np.argmax(p))
        prob_vec = p.tolist()
        if le is not None:
            label = le.inverse_transform([label_idx])[0]
        else:
            # assume order [CLEAR, MODERATE, HEAVY]
            mapping = {0:"CLEAR", 1:"MODERATE", 2:"HEAVY"}
            label = mapping.get(label_idx, str(label_idx))
        print(f"row={idx} Distance={last_dist:.2f} -> {label} prob_vec={prob_vec}")
    else:
        # binary/probability output case (sigmoid)
        prob = float(p[0])
        label = "HEAVY" if prob >= 0.5 else "CLEAR"
        print(f"row={idx} Distance={last_dist:.2f} -> {label} prob={prob:.3f}")
