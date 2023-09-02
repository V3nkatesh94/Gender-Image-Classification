'''Author Venkatesh'''
import numpy as np
from config import saveModelPath
import pandas as pd
import os
import pickle
import cv2
import streamlit as st
from skimage.filters import threshold_otsu

st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="wide"
)
# Title

st.title("Image Classification App")

# Upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
# image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)


with open(saveModelPath + "RandomForest.pkl", "rb") as model:
    linearModel = pickle.load(model)


if uploaded_image is not None:
    image = cv2.imread("D:\\image_classification\\test_image_folder\\test.jpg")
    image_width = st.slider("Select Image Width", 100, 1000)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (50, 50))
    threshold_value = threshold_otsu(gray)
    binary_image = gray > threshold_value
    st.image(
        uploaded_image,
        caption="Uploaded Image",
        use_column_width=True,
        width=image_width,
    )

    if st.button("Classify"):
        inp = binary_image.flatten()
        inp = inp.reshape(1, -1)
        with st.spinner("Classifying..."):
            predictions = linearModel.predict(inp)
        st.success("Classification complete!")
        print(predictions)
        if predictions[0] == 1:
            st.success(f"Predicted Target: Female")
        else:
            st.success(f"Predicted Target: Male")
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a pre-trained RandomForest model to classify images into different categories. "
    "The model was trained on the face dataset. Upload an image and click 'Classify' to get predictions."
)
