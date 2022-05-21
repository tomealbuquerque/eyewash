from pydantic import BaseModel

class CarDetectionResponse(BaseModel):
    bounding_box : list

class BrandModelDetection(BaseModel):
    prediction_brands: dict

class DirtyProbability(BaseModel):
    dirty_probability: int