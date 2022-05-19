# Imports
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np


def detect_objects(im: np.array):
  # Read image
  # im = cv2.imread(path_to_image)
  # Perform detection and get results
  #model = "yolov4"
  bbox, label, conf = cv.detect_common_objects(im ,model = "yolov4", confidence=0.4)
  # Draw bounding boxes over detected objects
  output_image = draw_bbox(im, bbox, label, conf)

  return bbox,label,conf


def filter_cars_detected(list_objects):
  boxes_car = []
  #Find label "CAR" and its bbox
  for idx, l in enumerate(list_objects[1]):
    if l == 'car' or l=='truck':
      # print(f"Detected object: {l} with confidence level of {c}\n")
      boxes_car.append(list_objects[0][idx])
  
  return boxes_car


def calculate_area_bbox(bbox:list):
  if len(bbox) != 4:
    raise Exception("Sorry, bouding box must contain 4 values")
  xmin,ymin,xmax,ymax = bbox
  #[xmin,ymin,xmax,ymax]
  area = (xmax - xmin) * (ymax - ymin)
  return area

def crop_image_bbox(image,bbox):
  xmin,ymin,xmax,ymax = bbox
  crop_image = image[xmin:xmax,ymin:ymax]
  return crop_image

def bounding_box_detection(im: np.array):
  #Detect objects
  res = detect_objects(im)

  #Filter by cars
  boxes_cars = filter_cars_detected(res)
  
  return boxes_cars

if __name__ == "__main__":
  bounding_box_detection()
