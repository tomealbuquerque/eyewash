import sys
from fastapi import APIRouter, status, HTTPException, UploadFile, File

from car_model_classifier.code.model_test import predict_car_model
from ..utils.read_image import read_imagefile


router = APIRouter(
    prefix = '/brand_model_detection',
    tags = ['Brand Model Detection']
)

sys.path.append('car_model_classifier')

@router.post('/', status_code = status.HTTP_201_CREATED)
async def prediction(file: UploadFile = File(...)):
    #integration        
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')

    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'Image must be jpg, jpeg or png format')

    im = read_imagefile(await file.read())
    prediction_model = predict_car_model(im)

    print(prediction_model)
    
    return prediction_model