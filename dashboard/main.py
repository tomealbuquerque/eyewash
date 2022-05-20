import streamlit as st
from PIL import Image
import requests
from requests.exceptions import HTTPError
from matplotlib import pyplot as plt

import numpy as np
from io import BytesIO
def read_imagefile(file) -> Image.Image:
    image = Image.open(file).convert('RGB')
    return np.array(image)

def load_image(image_file):
    img = Image.open(image_file)
    return img

BASE_URL = 'http://api:8002/'

import os

def make_request(endpoint: str, img_path: str):
    
    url = BASE_URL + endpoint

    headers = {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data'
    }

    files = {
        'file': open(img_path, 'rb')
    }

    response = requests.post(f"http://api:8002/{endpoint}/", files=files)

    st.write(response.json())

    return response


LOGO = read_imagefile('assets/logo.png')

with st.sidebar:
    st.image(LOGO)
    st.markdown('''---''')  # separator

    st.markdown('**Tab**')
    mechanism = st.radio(
        label='Select one of the following tabs:',
        options=['Main', 'Dirtyness Detection', "Car Make and Model", "Car Detection", "Decision Process"]
    )
    st.markdown('''---''')  # separator


if mechanism == 'Main':
    st.title("Eyewash AI Assessment")

    # Add project description
    # Use st.video to load a file containing the predictions for each frame (run this in a separate script)

    # Show the two images Tomé sent to discord, with the heatmaps (one of a clean car and one of a dirty car)
    # st.columns(2) with st.image 
    col1, col2 = st.columns(2)

    with col1:
        im1 = st.image(LOGO)
        st.write("Clean car")

    with col2:
        im2 = st.image(LOGO)
        st.write("Dirty car")

    # Add project team pictures / description


elif mechanism == "Dirtyness Detection":

    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    if image_file:
        image = read_imagefile(image_file)

        plt.imsave('uploaded.png' , image)
        st.image(image)

        res = make_request('dirtyness_level_detection', 'uploaded.png')

        activation_map = read_imagefile(os.path.join('..', res.json()['activation_map_path']))

        # TODO: Blend images like Tomé did
        st.image(activation_map)

        # Call API for dirtyness detection and bounding box detection.
        # Draw image with bounding box
        # requests.post('http://api:8080?get_bounding_box'...)


elif mechanism == 'Car Make and Model':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_ymm.png' , image)

        st.image(image)
        
        res = make_request('brand_model_detection', 'uploaded_ymm.png')


elif mechanism == "Car Detection":
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_car_detection.png' , image)

        st.image(image)
        
        res = make_request('car_detection', 'uploaded_car_detection.png')


elif mechanism == 'Decision Process':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_decision_process.png' , image)

        st.image(image)
        
        res = make_request('decision_process', 'uploaded_decision_process.png')