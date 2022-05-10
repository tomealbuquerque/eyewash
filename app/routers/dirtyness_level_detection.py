from fastapi import APIRouter, status, UploadFile, File

from .. import schemas
router = APIRouter(
    prefix = '/dirtyness_level_detection',
    tags = ['Dirtyness Level Detection']
)

@router.post('/', status_code = status.HTTP_201_CREATED, response_model = schemas.DirtyProbability)
async def prediction(file: UploadFile = File(...)):




    print('Hello World')

@router.get('/')
def root():
    return {'message': 'Galp_Hackaton_2022'}