import streamlit as st
from PIL import Image
import requests
from requests.exceptions import HTTPError
from matplotlib import pyplot as plt


def load_image(image_file):
    img = Image.open(image_file)
    return img

BASE_URL = 'http://api:8002/'

def make_request(endpoint :str, img_path:str):
    
    url = BASE_URL + endpoint

    headers = {
    'accept': 'application/json',
    # requests won't add a boundary if this header is set when you pass files=
    'Content-Type': 'multipart/form-data',
    }

    #files = {'file': open(f'{img_path}', 'rb'),}
    #response = requests.post(BASE_URL + endpoint , headers=headers, files=files)
    print("IMAGE PATH ", img_path)
    headers = {
    'accept': 'application/json',
    'Content-Type': 'multipart/form-data',
    }

    files = {
    'file': ('uploaded.png;type', open('uploaded.png;type', 'rb')),
    }

    response = requests.post('http://localhost:8002/car_detection/', headers=headers, files=files)
    """
    with open(img_path + ';type:image/png', 'rb') as f:
        response = requests.post(url, headers=headers, files={'file': f})
        print('Success!')
        print('GOT FROM API: ', response)
        return response
        #response = requests.post(url)
        # If the response was successful, no Exception will be raised
        #response.raise_for_status()
    """
    """
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  
    except Exception as err:
        print(f'Other error occurred: {err}')
    """
    return None
        
    
    




LOGO = load_image('assets/logo.png')

with st.sidebar:
    st.image(LOGO)
    st.markdown('''---''')  # separator

    st.markdown('**Tab**')
    mechanism = st.radio(
        label='Select one of the following tabs:',
        options=['Main', 'Dirtyness Detection', "Car Make and Model"]
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

    mechanism_2 = st.radio(
        label='Select one of the following tabs:',
        options=['Tab 1', 'Tab 2']
    )

    if mechanism_2 == 'Tab 1':
        st.write("Tab 1 code")

    if image_file:
        image = load_image(image_file)
        plt.imsave('uploaded.png' , image)
        st.image(image)

        st.write("Prediction xyz")

        res = make_request('car_detection','uploaded.png')
    
        st.write(res)
        # Call API for dirtyness detection and bounding box detection.
        # Draw image with bounding box
        # requests.post('http://api:8080?get_bounding_box'...)


elif mechanism == 'Car Make and Model':
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file:
        image = load_image(image_file)
        st.image(image)
        st.write("Prediction xyz")

        # Call API for make and model detection
        # requests.post(...)

