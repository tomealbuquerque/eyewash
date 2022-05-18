from fastapi import APIRouter

router = APIRouter(
    prefix = '/decision_process',
    tags = ['Decision Process']
)

@router.get('/')
def root():

    # recebe uma imagem
    # corre object detector e corta
    # corre modelo do Tomé
    # corre modelo do Tiago (make + model)
    # escreve em MongoDB

    return {'message': 'Galp_Hackaton_2022'}