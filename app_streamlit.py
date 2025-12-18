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
    return load_model("cnn-models.h5")

model = load_cnn_model()

# ================= PREPROCESS =================
def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (180, 180))

    img = cv2.GaussianBlur(img, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=1.0,
        tileGridSize=(16, 16)
    )
    img = clahe.apply(img)

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 180, 180, 1)
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
            prob = float(model.predict(processed)[0][0])

        if prob >= 0.6:
            label = "NOT FRACTURED"
            confidence = prob * 100
            st.success("🟢 Result: NOT FRACTURED")
        else:
            label = "FRACTURED"
            confidence = (1 - prob) * 100
            st.error("🔴 Result: FRACTURED")

        # ================= RESULT =================
        st.markdown("### 🧠 Prediction Result")
        colA, colB = st.columns(2)

        with colA:
            st.metric("Prediction Label", label)

        with colB:
            st.metric("Confidence", f"{confidence:.2f}%")
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
