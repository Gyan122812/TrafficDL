import matplotlib.pyplot as plt
import joblib

# Load the saved training history dictionary
history = joblib.load("training_history.pkl")

# Accuracy plot
plt.figure(figsize=(10,5))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("training_accuracy_curve.png")
plt.show()

# Loss plot
plt.figure(figsize=(10,5))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("training_loss_curve.png")
plt.show()
