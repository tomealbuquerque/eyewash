"""
# import modules
from bson import ObjectId
from sqlalchemy import TIMESTAMP
from pydantic import Field

from pydantic import BaseModel
#from sqlalchemy import TIMESTAMP

class PyObjectId(ObjectId):
    @classmethod
    def __get__validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type = 'string')

class Car(BaseModel):
    id: PyObjectId = Field(default_factory = PyObjectId, alias = '_id')
    created_at: TIMESTAMP
    brand: str = Field(...)
    model: str = Field(...)
    dirtyness_level: str = Field(...)
    decision: str = Field(...)
"""