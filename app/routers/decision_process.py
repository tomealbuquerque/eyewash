from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder



from .. import models, database

router = APIRouter(
    prefix = '/decision_process',
    tags = ['Decision Process']
)

from fastapi import APIRouter, UploadFile, HTTPException, status, File

from detect_cars.detect_cars import filter_cars_detected, detect_objects
from car_model_classifier.code.model_test import predict_car_model
from clean_dirty_cars_classifier.test import test_classifier

import sys
from PIL import Image

from ..utils.read_image import read_imagefile

sys.path.append('car_model_classifier')
sys.path.append('clean_dirty_cars_classifier')
sys.path.append('detect_cars')


@router.get('/')
def root():

    return {'message': 'Galp_Hackaton_2022'}

@router.post('/', status_code = status.HTTP_201_CREATED)
async def decision_process(file: UploadFile = File(...)):

    im = read_imagefile(await file.read())

    # Run object detector
    bbox, label, conf = detect_objects(im)

    # Get only the "car" objects
    boxes_car = filter_cars_detected((bbox, label, conf))

    # Create a list for detect cars
    cropped_cars=[]

    outcomes = []

    outcome = {}
    for bc in boxes_car:

        # Load image again
        image_full = Image.fromarray(im)
        image_full = image_full.convert('RGB')

        # Crop the area inside the bounding-box (car)
        image_cars = image_full.crop(bc)

        # Compute the dirtyness
        dirty_level = test_classifier(image_cars, model_path='clean_dirty_cars_classifier/baseline.pth')

        # Compute the Brand and Model of the Model
        car_model = predict_car_model(image_cars)

        outcome.update(dirty_level)
        outcome.update(car_model)
        outcomes.append(outcome)

        outcome = {}

    return outcomes
