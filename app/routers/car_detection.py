from fastapi import APIRouter, UploadFile, HTTPException, status, File

from detect_cars.detect_cars import bounding_box_detection, detect_objects
from ..utils.read_image import read_imagefile

import sys
router = APIRouter(
    prefix = '/car_detection',
    tags = ['Car Detection']
)

sys.path.append('detect_cars') 

@router.post('/', status_code = status.HTTP_201_CREATED)
async def detection(file: UploadFile = File(...)):
    # integration
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')

    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'image must be jpg, jpeg or png format')

    im = read_imagefile(await file.read())

    bounding_box = bounding_box_detection(im)

    return bounding_box
