import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import models, layers
from imblearn.over_sampling import RandomOverSampler
import joblib

# === Load data ===
X = np.load("X.npy")
y = np.load("y.npy")
X = X.reshape((X.shape[0], X.shape[1], 1))

print("Original class distribution:", np.bincount(y))

# === OVERSAMPLING TO FIX IMBALANCE ===
X_flat = X.reshape((X.shape[0], -1))

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_flat, y)

# reshape back to LSTM format
X_resampled = X_resampled.reshape((X_resampled.shape[0], X.shape[1], 1))

print("After oversampling:", np.bincount(y_resampled))

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, shuffle=True
)

# === Build LSTM Model ===
model = models.Sequential([
    layers.Input(shape=(X.shape[1], 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25, batch_size=16, verbose=1
)

# === Save model ===
model.save("traffic_lstm_model.h5")
model.save("traffic_lstm_model.keras")
print("✅ Model saved as both .h5 and .keras formats")

# === Save training history ===
joblib.dump(history.history, "training_history.pkl")
print("📁 training_history.pkl saved")

# === Plot Accuracy & Loss ===
plt.figure(figsize=(12,5))

# accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.savefig("training_accuracy_curve.png")

# loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig("training_loss_curve.png")

plt.tight_layout()
plt.show()
print("📁 Saved: accuracy & loss curves")

# === Confusion Matrix ===
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=['CLEAR', 'MODERATE', 'HEAVY'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (After Oversampling)")
plt.savefig("confusion_matrix.png")
plt.show()

print("📁 Saved: confusion_matrix.png")
