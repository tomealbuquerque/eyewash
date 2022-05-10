from fastapi import APIRouter, UploadFile, HTTPException, status, File

from .. import schemas

router = APIRouter(
    prefix = '/car_detection',
    tags = ['Car Detection']
)

@router.post('/', status_code = status.HTTP_201_CREATED)
def prediction(file: UploadFile = File(...)):
    # integration
    extension = file.filename.split('.')[-1] in ('.jpg', 'jpeg', 'png')
    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'image must be jpg, jpeg or png format')

    bounding_box = {'This is the bounding box'}
    return bounding_box