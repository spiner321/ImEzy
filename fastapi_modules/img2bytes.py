import io
import base64
from PIL import Image

def img_convert(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    
    return decoded