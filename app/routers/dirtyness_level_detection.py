from fastapi import APIRouter, HTTPException, status, UploadFile, File
from PIL import Image
from io import BytesIO
import os

from clean_dirty_cars_classifier.test_grad_cam import test_classifier_maps

router = APIRouter(
    prefix = '/dirtyness_level_detection',
    tags = ['Dirtyness Level Detection']
)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@router.post('/', status_code = status.HTTP_201_CREATED)
async def prediction(file: UploadFile = File(...)):
    # integration                        
    extension = file.filename.split('.')[-1] in ('.jpg', 'jpeg', 'png')
    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'image must be jpg, jpeg or png format')
    im = read_imagefile(await file.read())
    output = test_classifier_maps(im, model_path = '/Users/ctw02162/Personal/eyewash/clean_dirty_cars_classifier/baseline.pth')
    return output