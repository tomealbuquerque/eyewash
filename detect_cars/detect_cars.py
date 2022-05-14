# Imports
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


def detect_objects(path_to_image:str):
  # Read image
  im = cv2.imread(path_to_image)
  # Perform detection and get results
  bbox, label, conf = cv.detect_common_objects(im, model = "yolov4", confidence=0.5)
  # Draw bounding boxes over detected objects
  output_image = draw_bbox(im, bbox, label, conf)

  return zip(bbox,label,conf)


def filter_cars_detected(list_objects:list):
  boxes_car = []
  #Find label "CAR" and its bbox
  for b,l,c in list_objects:
    if l == 'car':
      print(f"Detected object: {l} with confidence level of {c}\n")
      boxes_car.append(b)
  
  return boxes_car


def calculate_area_bbox(bbox:list):
  if len(bbox) != 4:
    raise Exception("Sorry, bouding box must contain 4 values")
  xmin,ymin,xmax,ymax = bbox
  #[xmin,ymin,xmax,ymax]
  area = (xmax - xmin) * (ymax - ymin)
  return area

def crop_image_bbox(image,bbox:list):
  xmin,ymin,xmax,ymax = bbox
  crop_image = image[xmin:xmax,ymin:ymax]
  return crop_image_bbox


if __name__ == "__main__":
    path_img = 'dset_img_2.jpg'

    im = cv2.imread(path_img)

    #Detect objects
    res = detect_objects(path_img)


    #Filter by cars
    boxes_cars = filter_cars_detected(res)

    cropped = crop_image_bbox(im,boxes_cars[0])
    

    print(boxes_cars)

    areas = []
    for b in boxes_cars:
      a = calculate_area_bbox(b)
      print("Area ", a)
      areas.append(a)



"""
print("FINAL CAR BOXES LIST ", boxes_car)

# Show
print(int(label.count('car')))
plt.imshow(output_image)
plt.show()
plt.savefig('dset_img_results.png')
print('Number of cars in the image is '+ str(label.count('car')))
"""