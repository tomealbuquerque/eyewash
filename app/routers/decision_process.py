from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder



from .. import models, database

router = APIRouter(
    prefix = '/decision_process',
    tags = ['Decision Process']
)

@router.post('/')
async def root():
    car = jsonable_encoder(car)
    new_car = await database.db['cars'].insert_one(car)


    return {'message': 'Galp_Hackaton_2022'}