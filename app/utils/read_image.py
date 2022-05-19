#import modules
from PIL import Image
from io import BytesIO
import numpy as np

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert('RGB')
    return np.array(image)