import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from tensorflow.keras.models import load_model

WINDOW = 10

# Files
CSV_FILE = "traffic_data.csv"
SCALER_FILE = "scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
MODEL_FILE = "traffic_lstm_model.h5"

# Load everything
df = pd.read_csv(CSV_FILE)
df = df[df["distance_cm"] > 0].reset_index(drop=True)

scaler = joblib.load(SCALER_FILE)
model = load_model(MODEL_FILE)

label_encoder = None
try:
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
except:
    label_encoder = None

# Scale distances
scaled_vals = scaler.transform(df[["distance_cm"]])

# UI layout
st.title("Real-Time Traffic Congestion Prediction (IoT + LSTM)")
st.subheader("Powered by TinkerCAD Arduino + Deep Learning Model")

lcd = st.empty()
led_red = st.empty()
led_yellow = st.empty()
led_green = st.empty()
dist_box = st.empty()
prob_chart = st.empty()

buffer = []

# Real-time loop
for i in range(len(scaled_vals)):
    buffer.append(scaled_vals[i])
    if len(buffer) > WINDOW:
        buffer.pop(0)

    if len(buffer) == WINDOW:
        X = np.array(buffer).reshape(1, WINDOW, 1)
        pred = model.predict(X, verbose=0)[0]

        if pred.shape[-1] > 1:
            idx = int(np.argmax(pred))
            if label_encoder:
                label = label_encoder.inverse_transform([idx])[0]
            else:
                mapping = {0:"CLEAR", 1:"MODERATE", 2:"HEAVY"}
                label = mapping[idx]
            prob_vec = pred.tolist()
        else:
            # Binary model fallback
            prob = float(pred[0])
            label = "HEAVY" if prob >= 0.5 else "CLEAR"
            prob_vec = [1-prob, 0, prob]  

        # --------------- UI Updates ----------------

        dist = float(df.loc[i, "distance_cm"])

        dist_box.markdown(f"### Distance: **{dist:.2f} cm**")

        # Virtual LCD
        lcd.markdown(
            f"""
            <div style='font-size:30px; background:black; color:lime; padding:15px; border-radius:10px; text-align:center;'>
            ML Prediction: {label}
            </div>
            """, unsafe_allow_html=True
        )

        # LEDs
        def led_html(color_on):
            if color_on:
                return f"<div style='width:40px; height:40px; background:{color_on}; border-radius:50%; margin:10px auto;'></div>"
            else:
                return f"<div style='width:40px; height:40px; background:#333; border-radius:50%; margin:10px auto;'></div>"

        led_red.markdown("### Red LED" + led_html("red" if label=="HEAVY" else None), unsafe_allow_html=True)
        led_yellow.markdown("### Yellow LED" + led_html("yellow" if label=="MODERATE" else None), unsafe_allow_html=True)
        led_green.markdown("### Green LED" + led_html("green" if label=="CLEAR" else None), unsafe_allow_html=True)

        # Probability bar chart
        prob_chart.bar_chart({
            "CLEAR": [prob_vec[0]],
            "MODERATE": [prob_vec[1]],
            "HEAVY": [prob_vec[2]]
        })

        time.sleep(0.25)  # Real-time simulation speed
