# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:37:56 2022

@author: albu
"""

# System Imports
import sys
sys.path.append('..')

# Package Imports
import numpy as np
import cv2
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
from PIL import Image

# Project Imports
# import clean_dirty_cars_classifier
from car_model_classifier.code.model_test import predict_car_model
from clean_dirty_cars_classifier.test import test_classifier
from clean_dirty_cars_classifier.mymodels import *
from detect_cars import detect_objects,filter_cars_detected,calculate_area_bbox,crop_image_bbox


# Get an image
im = cv2.imread("unknown.png")

# Run object detector
bbox, label, conf = detect_objects(im)

# Get only the "car" objects
boxes_car = filter_cars_detected((bbox, label, conf))

# Create a list for detect cars
cropped_cars=[]


for bc in boxes_car:

    # Load image again
    image_full = Image.fromarray(im)
    image_full = image_full.convert('RGB')

    # Crop the area inside the bounding-box (car)
    image_cars = image_full.crop(bc)

    # Compute the dirtyness
    dirty_level = test_classifier(image_cars, model_path='baseline.pth')

    # Compute the Brand and Model of the Model
    car_model = predict_car_model(image_cars)

    # TODO: Write this information in MongoDB


    # Show image
    image_cars.show()
