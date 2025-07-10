import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os

# --- CONFIG ---
BASE_PATH = "C:\\Users\\acer\\Downloads\\archive\\Rice_Image_Dataset"
MODEL_PATH = os.path.join(BASE_PATH, "C:\\Users\\acer\\Downloads\\archive\\Rice_Image_Dataset\\model_cnn.keras")
CONF_MATRIX_PATH = os.path.join(BASE_PATH, "C:\\Users\\acer\\Downloads\\archive\\Rice_Image_Dataset\\confusion_matrix.csv")
CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# --- Load Resources ---
model = tf.keras.models.load_model(MODEL_PATH)
conf_df = pd.read_csv(CONF_MATRIX_PATH, index_col=0) if os.path.exists(CONF_MATRIX_PATH) else None

# --- UI ---
st.set_page_config(page_title="Rice Classifier", layout="wide")
st.sidebar.title("🔍 Navigation")
choice = st.sidebar.radio("Select section:", ["Prediction", "Model Metrics", "Rice Info"])
st.title("🌾 Rice Type Classification App")

# --- Prediction ---
if choice == "Prediction":
    st.subheader("📷 Upload an image for rice type prediction")
    uploaded_file = st.file_uploader("Choose a rice image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = cv2.resize(np.array(image), (100, 100)) / 255.0
        img = img.reshape(1, 100, 100, 3)

        pred_probs = model.predict(img)[0]
        pred_idx = np.argmax(pred_probs)
        st.success(f"✅ Predicted: **{CLASS_NAMES[pred_idx]}** ({pred_probs[pred_idx]:.2%} confidence)")

        st.subheader("🔎 Class probabilities")
        st.dataframe(pd.DataFrame({
            "Rice Type": CLASS_NAMES,
            "Confidence": [f"{p:.2%}" for p in pred_probs]
        }).set_index("Rice Type").style.highlight_max(axis=0))

# --- Model Metrics ---
elif choice == "Model Metrics":
    st.subheader("📊 Training Accuracy & Loss")
   
    if conf_df is not None:
        st.subheader("📌 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
        sns.heatmap(conf_df, annot=True, fmt="d", cmap="viridis")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        st.pyplot(fig_cm)
    else:
        st.warning("⚠️ confusion_matrix.csv not found")

# --- Rice Info ---
elif choice == "Rice Info":
    st.subheader("🌱 About Rice Types")
    info = {
        "Arborio": "Italian short-grain rice used in creamy risottos.",
        "Basmati": "Long-grain fragrant rice popular in Indian dishes.",
        "Ipsala": "Turkish rice ideal for pilaf.",
        "Jasmine": "Fragrant Thai long-grain rice.",
        "Karacadag": "Heirloom Turkish rice rich in minerals."
    }
    for k, v in info.items():
        st.markdown(f"### 🍚 {k}"); st.info(v)
