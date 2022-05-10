from fastapi import FastAPI

from .routers import car_detection, brand_model_detection, dirtyness_level_detection, decision_process

app = FastAPI()

app.include_router(car_detection.router)
app.include_router(brand_model_detection.router)
app.include_router(dirtyness_level_detection.router)
app.include_router(decision_process.router)

@app.get('/')
def root():
    return {'message': 'Galp_Hackaton_2022'}