from http.client import HTTPException
from fastapi import APIRouter, status, UploadFile, File

from .. import schemas
router = APIRouter(
    prefix = '/dirtyness_level_detection',
    tags = ['Dirtyness Level Detection']
)

@router.post('/', status_code = status.HTTP_201_CREATED, response_model = schemas.DirtyProbability)
async def prediction(file: UploadFile = File(...)):
    # integration                        
    extension = file.filename.split('.')[-1] in ('.jpg', 'jpeg', 'png')
    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'image must be jpg, jpeg or png format')