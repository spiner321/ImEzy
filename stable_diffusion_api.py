import json
import numpy as np
import io
import base64
from typing import Union, Optional
from PIL import Image
from datetime import datetime

from fastapi_modules import sd_model, inputs
from fastapi_modules.txt2img import txt2img
from fastapi_modules.img2bytes import img_convert
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# Pius wrote this
# from pydantic import BaseModel, Field
# from uuid import UUID
# from typing import Optional
# import os, io
# from starlette.responses import StreamingResponse
# import diffusion as df
# from PIL import Image
# import base64


# class Input(BaseModel):
#     id: UUID
#     prompt: str = Field(max_length=100)
#     negative_prompt: Optional[str] = Field(max_length=100)
#     prompt_style: Optional[str] = Field(max_length=30) # 'oil painting'
#     prompt_style2: Optional[str] = Field(max_length=30) # 'space'
#     steps: Optional[int] = Field(20, gt=0, lt=151) # 1 ~ 150
#     sampler_index: Optional[int] = Field(0, min_digit=0, max_digit=3)
#     restore_faces: Optional[bool] = Field(False)
#     tiling: Optional[bool] = Field(False)
#     n_iter: Optional[int] = Field(1, min_digit=1, max_digit=5)
#     batch_size: Optional[int] = Field(1, min_digit=1, max_digit=8)
#     cfg_scale: Optional[float] = Field(7.0, min_digit=0.1, max_digit=30.0)
#     denoising_strength = 0.75
#     seed: Optional[int]
#     subseed: Optional[int] = Field(-1, min_digit=-1, max_digit=1000)
#     subseed_strength: Optional[float] = Field(0.0, min_digit=0.0, max_digit=1.0)
#     seed_resize_from_h: Optional[int] = Field(0, min_digit=0, max_digit=2048)
#     seed_resize_from_w: Optional[int] = Field(0, min_digit=0, max_digit=2048)
#     seed_enable_extras: Optional[bool] = Field(False)
#     height: int = Field(512, min_digit=1, max_digit=2048)
#     width: int = Field(512, min_digit=1, max_digit=2048)
#     enable_hr: Optional[bool] = Field(False)
#     scale_latent: Optional[bool] = Field(True)
    
    
#     class Config:
#         schema_extra = {
#             "example": {
#                 "id": "a0a0a0a0-a0a0-a0a0-a0a0-a0a0a0a0a0a0",
#                 "prompt": "winter",
#                 "negative_prompt": "summer",
#                 "prompt_style": "oil painting",
#                 "prompt_style2": "space",
#                 "steps": 20,
#                 "sampler_index": 0,
#                 "restore_faces": False,
#                 "tiling": False,
#                 "n_iter": 1,
#                 "batch_size": 1,
#                 "cfg_scale": 7.0,
#                 "denoising_strength": 0.75,
#                 "seed": 0,
#                 "subseed": -1,
#                 "subseed_strength": 0.0,
#                 "seed_resize_from_h": 0,
#                 "seed_resize_from_w": 0,
#                 "seed_enable_extras": False,
#                 "height": 512,
#                 "width": 512,
#                 "enable_hr": False,
#                 "scale_latent": True
#             }
#         }
    
#     @staticmethod
#     def get_imgs(input):
#         imgs, info = txt2img(prompt=input.prompt, 
#                                 negative_prompt=input.negative_prompt, 
#                                 prompt_style=input.prompt_style, 
#                                 prompt_style2=input.prompt_style2, 
#                                 steps=input.steps,
#                                 sampler_index=input.sampler_index,
#                                 restore_faces=input.restore_faces,
#                                 tiling=input.tiling, 
#                                 n_iter=input.n_iter,
#                                 batch_size=input.batch_size,
#                                 cfg_scale=input.cfg_scale,
#                                 denoising_strength=input.denoising_strength,
#                                 seed=input.seed, 
#                                 subseed=input.subseed,
#                                 subseed_strength=input.subseed_strength,
#                                 seed_resize_from_h=input.seed_resize_from_h,
#                                 seed_resize_from_w=input.seed_resize_from_w,
#                                 seed_enable_extras=input.seed_enable_extras, 
#                                 height=input.height,
#                                 width=input.width,
#                                 scale_latent=input.scale_latent,
#                                 enable_hr=input.enable_hr,
#                             )
        
#         return imgs, info

# INPUTS = []

# app = FastAPI()

# @app.post('/pius')
# async def pius(input: Input):
#     print(input)
#     INPUTS.append(input)
#     return input

# @app.get('/pius')
# async def pius():
#     return INPUTS

# @app.post('/pius/diff/')
# async def pius_diff(input: Input):
#     images, info_js = Input.get_imgs(input)
#     info = json.loads(info_js)
#     print(info)
#     img_bytes = [x.tobytes() for x in images]
    
#     if not os.path.isdir(f'./static/'):
#         os.makedirs(f'./static/')
#     for idx, img in enumerate(images):
#         if idx == 0 and info["batch_size"] > 1:
#             img.save(f'./static/{info["all_seeds"][0]}-{info["all_prompts"][0]}_grid.png')    
#         img.save(f'./static/{info["all_seeds"][idx-1]}-{info["all_prompts"][idx-1]}.png')
    
#     def img_convert(img):
#         imgByteArr = io.BytesIO()
#         img.save(imgByteArr, format='PNG')
#         imgByteArr = imgByteArr.getvalue()
#         encoded = base64.b64encode(imgByteArr)
#         decoded = encoded.decode('ascii')
#         return decoded
    
#     img_list = []
#     for img in images:
#         img_list.append(img_convert(img))

#     return JSONResponse(content={"images": img_list, "info": info_js})

app = FastAPI()

info_list = []
@app.post('/txt2img_api/')
async def txt2img_api(param: inputs.txt2img_inputs,):  
    
    images, info_js = txt2img(prompt=param.prompt, 
                              negative_prompt=param.negative_prompt, 
                              prompt_style=param.prompt_style, 
                              prompt_style2=param.prompt_style2, 
                              steps=param.steps, 
                              sampler_index=param.sampler_index,
                              restore_faces=param.restore_faces, 
                              tiling=param.tiling, 
                              n_iter=param.n_iter, 
                              batch_size=param.batch_size, 
                              cfg_scale=param.cfg_scale, 
                              denoising_strength=param.denoising_strength, 
                              seed=param.seed,
                              subseed=param.subseed, 
                              subseed_strength=param.subseed_strength, 
                              seed_resize_from_h=param.seed_resize_from_h, 
                              seed_resize_from_w=param.seed_resize_from_w, 
                              seed_enable_extras=param.seed_enable_extras, 
                              height=param.height, 
                              width=param.width, 
                              scale_latent=param.scale_latent, 
                              enable_hr=param.enable_hr)
    
    dt = datetime.now().strftime('%y%m%d_%H%M')
    
    # image_bytes = images[0].tobytes()
    # image_bytes = image_bytes.decode('latin1')
    image_bytes = [img_convert(image) for image in images[1:]]
    
    images_js = {'username': inputs.username,
                 'uuid': param.uuid,
                 'image_bytes': image_bytes}
    
    file_path = f'/data/kimgh/ImEzy/stable-diffusion-flask/outputs/txt2img-json/{dt}_{inputs.username}_{param.uuid}.json'
    with open(file_path, 'w') as f:
        json.dump(images_js, f)
    
    info = {'uuid': param.uuid,
            'datetime': dt}
    info.update(json.loads(info_js))
    # info.update({'image_bytes': image_bytes})
    # image_array = np.asarray(images[0])
    # info.update({'image_bytes': image_array})
    
    info_list.append(info)

    return info


@app.get('/txt2img_api/{uuid}')
async def txt2img_js(uuid: str):
    path = 
    
    # # 실시간 info_js 정보 화면에 띄우기
    # if uuid == info_list[0]['uuid']:
    #     return info_list[0]