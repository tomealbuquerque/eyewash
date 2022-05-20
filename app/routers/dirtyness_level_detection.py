from fastapi import APIRouter, HTTPException, status, UploadFile, File
from PIL import Image
from io import BytesIO

from clean_dirty_cars_classifier.test_grad_cam import test_classifier_maps
from clean_dirty_cars_classifier.test import test_classifier

from ..utils.read_image import read_imagefile

# from clean_dirty_cars_classifier.test import test_classifier
import os
import sys

sys.path.append("clean_dirty_cars_classifier")

router = APIRouter(
    prefix="/dirtyness_level_detection", tags=["Dirtyness Level Detection"]
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def prediction(file: UploadFile = File(...)):
    # integration
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")

    if not extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image must be jpg, jpeg or png format",
        )

    im = read_imagefile(await file.read())

    output = test_classifier_maps(
        im, model_path="clean_dirty_cars_classifier/baseline.pth"
    )
    # output = test_classifier(im, model_path = 'clean_dirty_cars_classifier/baseline.pth')

    return output
