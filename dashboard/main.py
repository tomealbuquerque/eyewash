# Imports
import os
import streamlit as st
from PIL import Image
import requests
from requests.exceptions import HTTPError
from matplotlib import pyplot as plt
import json
import cv2
import random
import numpy as np
from io import BytesIO



def read_imagefile(file) -> Image.Image:
    image = Image.open(file).convert('RGB')
    return np.array(image)



def load_image(image_file):
    img = Image.open(image_file)
    return img

BASE_URL = 'http://api:8002/'



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

    # st.write(response.json())

    

    return response


LOGO = read_imagefile('assets/logo.png')

with st.sidebar:
    st.image(LOGO)
    st.markdown('''---''')  # separator

    st.markdown('**Tab**')
    mechanism = st.radio(
        label='Select one of the following tabs:',
        options=['Main', 'Car Detection', 'Dirtiness Level Detection', 'Car Model Detection', 'Decision Process Pipeline']
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
        st.write("Clean car")
        st.image(read_imagefile('assets/clean.png'))

    with col2:
        st.write("Dirty car")
        st.image(read_imagefile('assets/dirty.png'))
    
    st.write("Video showing project features:  \n")
    video_file = open('assets/heatmap_taxi.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes) 



elif mechanism == 'Car Detection':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_car_detection.png' , image)

        st.image(image)
        
        res = make_request('car_detection', 'uploaded_car_detection.png')
        

        loaded_response = json.loads(res.text)
        st.write("In the uploaded image were found: %s cars" % len(loaded_response))
        bounded_image= image
        for bbox in loaded_response:
            xmin,ymin,xmax,ymax=bbox
            st.write("Car found on:  \n XMin: %s  \n Xmax: %s  \n YMin: %s  \n  YMax: %s" % (xmin,xmax,ymin,ymax))
            color = tuple(np.random.random(size=3) * 256)
            cv2.rectangle(bounded_image , (xmin, ymin), (xmax, ymax), color, 2)
        st.image(bounded_image)



elif mechanism == 'Dirtiness Level Detection':

    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    if image_file:
        image = read_imagefile(image_file)

        plt.imsave('uploaded.png' , image)
        st.image(image)

        res = make_request('dirtyness_level_detection', 'uploaded.png')

        st.write(res.json())

        activation_map = read_imagefile(os.path.join('..', res.json()['activation_map_path']))

        # TODO: Blend images like Tomé did
        st.image(activation_map)

        # Call API for dirtyness detection and bounding box detection.
        # Draw image with bounding box
        # requests.post('http://api:8080?get_bounding_box'...)



elif mechanism == 'Car Model Detection':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_ymm.png' , image)

        st.image(image)
        
        res = make_request('brand_model_detection', 'uploaded_ymm.png')

        st.write(res.json())

    

elif mechanism == 'Decision Process Pipeline':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:

        image = read_imagefile(image_file)
        plt.imsave('uploaded_decision_process.png' , image)

        st.image(image)
        
        res = make_request('decision_process', 'uploaded_decision_process.png')

        content = res.json()
        for c in content:
            st.write("Vehicle found: Brand, Model and Year: %s" % (c['predicted_class_ymm']))
            st.write("Probability of being such model: %s" % c['probability_ymm'])
            st.write("Dirtiness condition: %s " % c['pred_class_dirty'])
            st.write("Probability of being dirty: %s" % c['probability_dirty'])
            st.write("Recommended clean program: %s" % c['recommended_program'])

            st.write("Check below the heatmap for the dirtiness found on vehicle:  \n")
            img_file = read_imagefile(os.path.join('..',c['activation_map_path']))
            st.image(img_file)
