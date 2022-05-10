from fastapi import APIRouter

router = APIRouter(
    prefix = '/decision_process',
    tags = ['Decision Process']
)

@router.get('/')
def root():
    return {'message': 'Galp_Hackaton_2022'}