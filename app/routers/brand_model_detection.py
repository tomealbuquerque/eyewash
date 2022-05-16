from fastapi import APIRouter, status, HTTPException, UploadFile, File

from .. import schemas

router = APIRouter(
    prefix = '/brand_model_detection',
    tags = ['Brand Model Detection']
)

@router.post('/', status_code = status.HTTP_201_CREATED, response_model = schemas.BrandModelDetection)
async def prediction(file: UploadFile = File(...)):
    #integration
    if not file:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND,
                            detail = 'no upload file sent')
                        
    extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
    if not extension:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, 
                            detail = 'Image must be jpg, jpeg or png format')
    
    prediction_brands = {'Here comes the output'}
    return prediction_brands