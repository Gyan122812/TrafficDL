import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# === Load your dataset ===
df = pd.read_csv("traffic_data.csv")

# 1. Clean up
df = df.dropna()
df = df[df["distance_cm"] > 0]       # remove invalid -1 readings
df = df.reset_index(drop=True)

# 2. Encode labels (CLEAR=0, MODERATE=1, HEAVY=2)
encoder = LabelEncoder()
df["label_encoded"] = encoder.fit_transform(df["label"])
joblib.dump(encoder, "label_encoder.pkl")

# 3. Scale the distance feature
scaler = MinMaxScaler()
df["dist_scaled"] = scaler.fit_transform(df[["distance_cm"]])
joblib.dump(scaler, "scaler.pkl")

# 4. Create time-window sequences for LSTM
WINDOW = 10   # number of past readings per sample
X, y = [], []
for i in range(len(df) - WINDOW):
    X.append(df["dist_scaled"].iloc[i:i+WINDOW].values)
    y.append(df["label_encoded"].iloc[i+WINDOW])  # next label

X, y = np.array(X), np.array(y)

print("X shape:", X.shape, " y shape:", y.shape)
np.save("X.npy", X)
np.save("y.npy", y)
