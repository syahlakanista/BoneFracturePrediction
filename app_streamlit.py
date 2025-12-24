import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore

# ================= CONFIG =================
st.set_page_config(
    page_title="X-Ray Fracture Detection",
    page_icon="🦴",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align: center;'>🦴 X-Ray Fracture Detection</h1>
    <p style='text-align: center; color: gray;'>
    CNN-based fracture prediction simulation using X-Ray images
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_cnn_model():
    return load_model("cnn-model.h5")

model = load_cnn_model()

# ================= PREPROCESS =================
def preprocess(img):
    # ERROR 1: Resize tidak konsisten (224 vs 180)
    # Training Anda pakai 180x180, jadi harus 180
    img = np.array(img)
    img = cv2.resize(img, (224, 224)) 

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    img = clahe.apply(img)

    img = img.astype("float32") / 255.0
    
    img = img.reshape(1, 224, 224, 1)
    return img

# ================= UPLOAD SECTION =================
st.subheader("📤 Upload X-Ray Image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")

    st.markdown("### 🖼️ Uploaded Image")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, width=350)

    st.markdown("---")

    # ================= PREDICT =================
    if st.button("🔍 Run Prediction", use_container_width=True):
        with st.spinner("Processing image and running CNN model..."):
            processed = preprocess(img)
            
            predictions = model.predict(processed, verbose=0)
            raw_prob = float(predictions[0][0])
            
            prob_not_fractured = raw_prob
            prob_fractured = 1.0 - prob_not_fractured

        if prob_fractured >= 0.5: 
            label = "FRACTURED"
            confidence = prob_fractured * 100
            st.error("🔴 Result: FRACTURED")
        else:
            label = "NOT FRACTURED"
            confidence = prob_not_fractured * 100
            st.success("🟢 Result: NOT FRACTURED")

        # ================= RESULT =================
        st.markdown("### 🧠 Prediction Result")
        colA, colB = st.columns(2)

        with colA:
            st.metric("Prediction Label", label)

        with colB:
            st.metric("Confidence", f"{confidence:.2f}%")
            
        st.markdown("#### 📊 Detailed Probabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Fractured:** {prob_fractured*100:.1f}%")
        with col2:
            st.write(f"**Not Fractured:** {prob_not_fractured*100:.1f}%")
            
        if confidence < 70:
            st.info("⚠️ Prediction confidence is low. Result should be interpreted carefully.")


# ================= FOOTNOTE =================
st.markdown(
"""
    <hr>
    <p style='text-align: center; font-size: 12px; color: gray;'>
    This website is a simulation for academic research purposes only.  
    It is not intended for clinical diagnosis.
    </p>
    """,
    unsafe_allow_html=True
)