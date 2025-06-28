import streamlit as st
import torch
from PIL import Image
import os

# Title and upload UI
st.title("🚦 Speed Sign Detector")
uploaded_file = st.file_uploader("Upload a road race image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load YOLOv5 model (assumes best.pt is in the same folder)
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    # Run detection
    results = model(image)
    results.render()  # draws bounding boxes on results.ims[0]

    # Display result
    st.image(Image.fromarray(results.ims[0]), caption="Detected Output", use_column_width=True)
