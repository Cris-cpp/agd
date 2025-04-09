import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

# Load models
model = load_model('Age_Sex_Detection.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect and crop face
def recognize_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]
    cropped_face = cv2.resize(cropped_face, (128, 128))
    return cropped_face

# Streamlit UI
st.set_page_config(page_title="Age & Gender Detector", layout="centered")
st.title("🧠 Age & Gender Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cropped_face = recognize_and_crop_face(image_bgr)

    if cropped_face is None:
        st.error("No face detected. Please try another image.")
    else:
        input_img = np.array(cropped_face) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        pred = model.predict(input_img)
        age = int(np.round(pred[1][0]))
        gender = "Male" if pred[0][0] > 0.5 else "Female"

        st.success(f"**Detected Gender:** {gender}")
        st.success(f"**Estimated Age:** {age}")
