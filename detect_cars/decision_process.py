# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:37:56 2022

@author: albu
"""

import cv2
from detect_cars import detect_objects,filter_cars_detected,calculate_area_bbox,crop_image_bbox
import matplotlib.pyplot as plt
import numpy as np
from cvlib.object_detection import draw_bbox
from PIL import Image

import sys
sys.path.append('..')

# import clean_dirty_cars_classifier
from clean_dirty_cars_classifier.test import test_classifier
from clean_dirty_cars_classifier.mymodels import *


im = cv2.imread("unknown.png")

bbox,label,conf = detect_objects(im)

boxes_car = filter_cars_detected((bbox,label,conf))

cropped_cars=[]
for bc in boxes_car:
    image_full = Image.fromarray(im)
    image_cars= image_full.crop(bc)
    dirty_level=test_classifier(image_cars, model_path='baseline.pth')
    image_cars.show()
   
    
   
    

    # recebe uma imagem
    # corre object detector e corta
    # corre modelo do Tomé
    # corre modelo do Tiago (make + model)
    # escreve em MongoDB