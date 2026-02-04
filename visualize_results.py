import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
import joblib

# === Load data ===
X = np.load("X.npy")
y = np.load("y.npy")
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Load model ===
model = load_model("traffic_lstm_model.h5")
print("Model loaded.")

# === Predict on entire dataset (or only test set if you prefer) ===
y_pred = np.argmax(model.predict(X), axis=1)

# === Raw Confusion Matrix ===
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CLEAR', 'MODERATE', 'HEAVY'])

plt.figure(figsize=(7,6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Raw)")
plt.savefig("confusion_matrix.png")
print("📁 Saved: confusion_matrix.png")

# === Normalized Confusion Matrix ===
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                   display_labels=['CLEAR', 'MODERATE', 'HEAVY'])

plt.figure(figsize=(7,6))
disp_norm.plot(cmap='Blues', values_format='.2f')
plt.title("Confusion Matrix (Normalized)")
plt.savefig("confusion_matrix_normalized.png")
print("📁 Saved: confusion_matrix_normalized.png")

plt.show()
