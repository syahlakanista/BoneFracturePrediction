# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from tensorflow.keras.models import load_model # type: ignore

# # ================= CONFIG =================
# st.set_page_config(
#     page_title="X-Ray Fracture Detector",
#     layout="centered"
# )

# st.title("🦴 X-Ray Fracture Detection (CNN)")
# # st.write("Upload X-Ray image to predict fracture condition.")

# # ================= LOAD MODEL =================
# @st.cache_resource
# def load_cnn_model():
#     return load_model("cnn-models.h5")

# model = load_cnn_model()

# # ================= PREPROCESS (SAMA DENGAN TRAINING) =================
# def preprocess(img):
#     img = np.array(img)                     # PIL → NumPy
#     img = cv2.resize(img, (180, 180))       # resize

#     clahe = cv2.createCLAHE(
#         clipLimit=2.0,
#         tileGridSize=(8, 8)
#     )
#     img = clahe.apply(img)                  # CLAHE

#     img = img.astype("float32") / 255.0     # normalize
#     img = img.reshape(1, 180, 180, 1)       # (batch, h, w, c)
#     return img

# # ================= UI =================
# uploaded_file = st.file_uploader(
#     "Upload X-Ray image to predict fracture condition",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("L")

#     # 👉 CENTER IMAGE
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.image(img, caption="Uploaded X-Ray Image", width=400)

#     if st.button("🔍 Predict"):
#         processed = preprocess(img)

#         prob = float(model.predict(processed)[0][0])

#         # class mapping:
#         # 0 = fractured
#         # 1 = not fractured
#         if prob >= 0.3:
#             label = "NOT FRACTURED"
#             confidence = prob * 100
#         else:
#             label = "FRACTURED"
#             confidence = (1 - prob) * 100

#         st.subheader("🧠 Prediction Result")
#         st.write(f"**Label:** {label}")
#         st.write(f"**Confidence:** {confidence:.2f}%")
#         st.write(f"**Raw probability (class 1):** {prob:.4f}")

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

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

        if prob >= 0.3:
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
