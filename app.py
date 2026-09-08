import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "best_vgg16_brain_tumor"   # update path if needed
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # must match training order

CLASS_INFO = {
    "glioma": "Glioma Tumor detected",
    "meningioma": "Meningioma Tumor detected",
    "pituitary": "Pituitary Tumor detected",
    "notumor": "No Tumor detected",
}

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain MRI Tumor Classification")
st.write("Upload a brain MRI scan to classify it as **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.")

# ----------------------------
# Load model (cached so it only loads once per session)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

with st.spinner("Loading model..."):
    model = load_model()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------
# Upload & predict
# ----------------------------
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI scan", width="stretch")

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            arr = preprocess_image(image)
            preds = model.predict(arr)[0]
            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])

        st.markdown("---")
        if pred_class == "notumor":
            st.success(f"✅ {CLASS_INFO[pred_class]}  \nConfidence: **{confidence:.2%}**")
        else:
            st.error(f"⚠️ {CLASS_INFO[pred_class]}  \nConfidence: **{confidence:.2%}**")

        st.subheader("Class Probabilities")
        for cls, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
            st.write(f"{cls}: {prob:.2%}")
            st.progress(float(prob))

        st.caption("⚠️ This tool is for educational/demonstration purposes only and is not a substitute for professional medical diagnosis.")
else:
    st.info("Please upload an MRI image to get started.")