import json
from fastapi import APIRouter, UploadFile, HTTPException, status, File

from .. import schemas

router = APIRouter(
    prefix = '/car_detection',
    tags = ['Car Detection']
)

@router.post('/', status_code = status.HTTP_201_CREATED, response_model = schemas.CarDetectionResponse)
async def prediction(file: UploadFile = File(...)):
    # integration
    extension = file.filename.split('.')[-1] in ('.jpg', 'jpeg', 'png')
    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'Image must be jpg, jpeg or png format')
                        
    #bounding_box = {'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0}
    #return json.dumps(bounding_box)
    bounding_box = []
    return bounding_box