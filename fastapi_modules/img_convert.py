import io
import base64
from PIL import Image


def img2bytes(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    
    return decoded


def bytes2img(bytes):
    image_bytes = base64.b64decode(bytes)
    img = io.BytesIO(image_bytes)

    return Image.open(img)