import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 🧠 Load trained CNN model
model = tf.keras.models.load_model("C:\\Users\\acer\\Downloads\\archive\\Rice_Image_Dataset\\model_cnn.keras")

# 🏷️ Class labels (order same as training)
class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# 🎨 Streamlit UI setup
st.set_page_config(page_title="🌾 Rice Classifier (CNN)", layout="centered")
st.title("🌾 Rice Type Classifier using CNN")
st.markdown("Upload a rice grain image. The model will classify it and evaluate test accuracy with confusion matrix.")

# 📤 Image upload
file = st.file_uploader("📤 Upload a rice image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file:
    # 🖼️ Load and show image
    image = Image.open(file).convert("RGB").resize((100, 100))
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # 🔄 Preprocess image
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🧠 Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    confidence = prediction[0][pred_index]
    predicted_class = class_names[pred_index]

    # ✅ Show result
    if confidence < 0.6:
        st.error("⚠️ Model is not confident. Please upload a clear rice grain image.")
    else:
        st.success(f"✅ Predicted Rice Type: **{predicted_class}**")
        st.info(f"🔢 Confidence: **{(confidence * 100):.2f}%**")

        # 📊 Confidence Bar Graph
        st.subheader("📊 Class Confidence Scores")
        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0], color="skyblue")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# 📈 Accuracy + Confusion Matrix
st.markdown("---")
st.subheader("📈 Evaluate Accuracy & Confusion Matrix on Test Set")

if st.button("🔍 Evaluate Model"):
    try:
        # Load saved test set
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"✅ Model Accuracy on Test Set: **{acc * 100:.2f}%**")

        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        # Confusion Matrix
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")