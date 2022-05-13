#import modules
from PIL import Image
from io import BytesIO


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image