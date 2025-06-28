import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

st.set_page_config(page_title="Speed Sign Detector", layout="centered")
st.title("🚦 Speed Sign Detector")

uploaded_file = st.file_uploader("Upload a road race image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)

    # Run prediction
    results = model(image)
    results.render()

    st.image(Image.fromarray(results.ims[0]), caption="Detected Output", use_column_width=True)
