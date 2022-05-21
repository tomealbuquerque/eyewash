# import modules

from fastapi import APIRouter, UploadFile, HTTPException, status, File

from detect_cars.detect_cars import filter_cars_detected, detect_objects
from car_model_classifier.code.model_test import predict_car_model
from clean_dirty_cars_classifier.test_grad_cam import test_classifier_maps

import sys
from datetime import datetime
from PIL import Image

from ..utils.read_image import read_imagefile
from ..utils import database

router = APIRouter(
    prefix = '/decision_process',
    tags = ['Decision Process']
)

sys.path.append('car_model_classifier')
sys.path.append('clean_dirty_cars_classifier')
sys.path.append('detect_cars')

@router.post('/', status_code = status.HTTP_201_CREATED)
async def decision_process(file: UploadFile = File(...)):
    # integration
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')

    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST,
                            detail = 'image must be jpg, jpeg or png format')

    im = read_imagefile(await file.read())

    # Run object detector
    bbox, label, conf = detect_objects(im)

    # Get only the "car" objects
    boxes_car = filter_cars_detected((bbox, label, conf))

    # Create a list for detect cars
    cropped_cars = []

    outcomes = []

    outcome = {}
    for bc in boxes_car:

        # Load image again
        image_full = Image.fromarray(im)
        image_full = image_full.convert('RGB')

        # Crop the area inside the bounding-box (car)
        image_cars = image_full.crop(bc)

        # Compute the dirtyness
        dirty_level = test_classifier_maps(image_cars, model_path='clean_dirty_cars_classifier/baseline.pth')

        # Program prediction
        # between 0 - 0.2 : None
        # between 0.2 - 0.4 - Simples
        # Between 0.4 - 0.6 - Super
        # Between 0.6 - 0.8 - Especial
        # Between 0.8 - 1.0 - Extra

        dirty_proba = dirty_level['probability_dirty']
        if dirty_proba < 0.2:
            recommended_program = 'none'
        elif (dirty_proba > 0.2) and (dirty_proba < 0.4):
            recommended_program = 'simples'
        elif (dirty_proba > 0.4) and (dirty_proba < 0.6):
            recommended_program = 'super'
        elif (dirty_proba > 0.6) and (dirty_proba < 0.8):
            recommended_program = 'especial'
        elif (dirty_proba > 0.8):
            recommended_program = 'extra'
        
        dirty_level['recommended_program'] = recommended_program

        # Compute the Brand and Model of the Model
        car_model = predict_car_model(image_cars)

        outcome.update(dirty_level)
        outcome.update(car_model)
        outcome.update({'bbox': bc})

        outcomes.append(outcome)
    
        outcome['timestamp'] = datetime.now()

        # save on the DB
        database.db['cars'].insert_one(outcome)

        outcome.pop('_id')

        outcome = {}

    return outcomes