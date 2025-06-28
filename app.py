import streamlit as st
from PIL import Image
import torch
import os

st.set_page_config(page_title="Terrain & Marker Detector", layout="centered")
st.title("🏁 Terrain & Visual Marker Detector")

uploaded_file = st.file_uploader("Upload a race image (e.g., road, trail, sign markers)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    image_path = "temp_image.jpg"
    image.convert("RGB").save(image_path, format="JPEG")

    # Load model (make sure 'best.pt' is uploaded to repo root)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

    # Inference
    results = model(image_path)

    # Display results as image
    results.save()  # saves to 'runs/detect/exp/'

    # Show prediction image
    pred_image_path = "runs/detect/exp/image0.jpg"
    if os.path.exists(pred_image_path):
        st.image(Image.open(pred_image_path), caption="Detected Markers", use_column_width=True)
    else:
        st.error("Prediction image not found.")
