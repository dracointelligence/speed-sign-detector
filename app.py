import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

model = load_model()

st.title("🏁 Speed Sign Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Original", use_column_width=True)

    results = model(img)
    results.render()

    st.image(np.array(results.ims[0]), caption="Detected", use_column_width=True)
