import json
import uuid
import os
from datetime import datetime

# fastapi
from fastapi import FastAPI, Depends, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder

# model
from fastapi_modules import sd_model, inputs
from fastapi_modules.txt2img import txt2img
from fastapi_modules.img2img import img2img
from fastapi_modules.img_convert import img2bytes, bytes2img

# db
# from fastapi_modules import db_models
from fastapi_modules.database import engine, SessionLocal, Base, Input_Info_DB, insert_db
from sqlalchemy.orm import Session

# create DB
Base.metadata.create_all(bind=engine)

def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


app = FastAPI()

txt2img_info = []
img2img_info = []

@app.post('/txt2img_api/')
async def txt2img_api(param: inputs.txt2img_inputs, db: Session=Depends(get_db)):  
    uuid_ = str(uuid.uuid4()) # 버전은 1~4까지 있음 / 버전1 : timestamp기준, 버전4 : 완전랜덤

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
    
    dt_now_utc = datetime.utcnow()
    dt_str_js = dt_now_utc.strftime('%y%m%d_%H%M')
    # dt_db = datetime.strptime(str(dt_now_utc)[:-7], '%Y-%m-%d %H:%M:%S')
    
    # image_bytes = images[0].tobytes()
    # image_bytes = image_bytes.decode('latin1')
    image_bytes = [img2bytes(image) for image in images[1:]]
    
    images_js = {'username': inputs.username,
                 'uuid': uuid_,
                 'image_bytes': image_bytes}
    
    file_path = f'/data/kimgh/ImEzy/stable-diffusion-flask/outputs/txt2img-json/{dt_str_js}_{inputs.username}_{uuid_}.json'
    with open(file_path, 'w') as f:
        json.dump(images_js, f)
    
    info = {'uuid': uuid_,
            'datetime': dt_str_js}
    info.update(json.loads(info_js))
    # info.update({'image_bytes': image_bytes})
    # image_array = np.asarray(images[0])
    # info.update({'image_bytes': image_array})
    
    # info 내용을 get url에 띄우는 코드
    global txt2img_info
    txt2img_info = []
    txt2img_info.append(info)

    result = info.copy()
    result.update({'image_bytes': image_bytes})

    input_db = Input_Info_DB()
    input_db = insert_db(input_db, inputs.username, uuid_, info, dt_now_utc, 't')
    db.add(input_db)
    db.commit()

    return result


@app.get('/txt2img_api/{uuid}')
async def txt2img_js(uuid: str):
    if uuid == txt2img_info[0]['uuid']:
        path = '/data/kimgh/ImEzy/stable-diffusion-flask/outputs/txt2img-json/'

        txt2img_ls = os.listdir(path)
        txt2img_js = [js for js in txt2img_ls if uuid in js][0]

        with open(path+txt2img_js, 'r') as f:
            js = json.load(f)
        
        return js

    # 실시간 info_js 정보 화면에 띄우기
    # if uuid == txt2img_info[0]['uuid']:
    #     return txt2img_info[0]


@app.post('/img2img_api/')
async def img2img_api(param: inputs.img2img_inputs, db: Session=Depends(get_db)):
    uuid_ = str(uuid.uuid4())

    images, info_js = img2img(mode=param.mode,
                              prompt=param.prompt, 
                              negative_prompt=param.negative_prompt, 
                              prompt_style=param.prompt_style, 
                              prompt_style2=param.prompt_style2, 
                              init_img=bytes2img(param.init_img), 
                              init_img_with_mask=param.init_img_with_mask,
                              init_img_inpaint=param.init_img_inpaint, 
                              init_mask_inpaint=param.init_mask_inpaint, 
                              mask_mode=param.mask_mode, 
                              steps=param.steps, 
                              sampler_index=param.sampler_index, 
                              mask_blur=param.mask_blur, 
                              inpainting_fill=param.inpainting_fill,
                              restore_faces=param.restore_faces, 
                              tiling=param.tiling, 
                              n_iter=param.n_iter, 
                              batch_size=param.batch_size, 
                              cfg_scale=param.cfg_scale, 
                              denoising_strength=param.denoising_strength, 
                              seed=param.seed, subseed=param.subseed,
                              subseed_strength=param.subseed_strength, 
                              seed_resize_from_h=param.seed_resize_from_h, 
                              seed_resize_from_w=param.seed_resize_from_w, 
                              seed_enable_extras=param.seed_enable_extras, 
                              height=param.height, 
                              width=param.width,
                              resize_mode=param.resize_mode, 
                              inpaint_full_res=param.inpaint_full_res, 
                              inpaint_full_res_padding=param.inpaint_full_res_padding, 
                              inpainting_mask_invert=param.inpainting_mask_invert,
                              img2img_batch_input_dir=param.img2img_batch_input_dir, 
                              img2img_batch_output_dir=param.img2img_batch_output_dir)

    dt_now_utc = datetime.utcnow()
    dt_str_js = dt_now_utc.strftime('%y%m%d_%H%M')

    image_bytes = [img2bytes(image) for image in images[1:]]
    
    images_js = {'username': inputs.username,
                 'uuid': uuid_,
                 'image_bytes': image_bytes}
    
    file_path = f'/data/kimgh/ImEzy/stable-diffusion-flask/outputs/img2img-json/{dt_str_js}_{inputs.username}_{uuid_}.json'
    with open(file_path, 'w') as f:
        json.dump(images_js, f)
    
    info = {'uuid': uuid_,
            'datetime': dt_str_js}
    info.update(json.loads(info_js))

    global img2img_info
    img2img_info = []
    img2img_info.append(info)

    result = info.copy()
    result.update({'image_bytes': image_bytes})

    input_db = Input_Info_DB()
    input_db = insert_db(input_db, inputs.username, uuid_, info, dt_now_utc, 'i')
    db.add(input_db)
    db.commit()

    return result


@app.get('/img2img_api/{uuid}')
async def img2img_js(uuid: str):
    if uuid == img2img_info[0]['uuid']:
        path = '/data/kimgh/ImEzy/stable-diffusion-flask/outputs/img2img-json/'

        img2img_ls = os.listdir(path)
        img2img_js = [js for js in img2img_ls if uuid in js][0]

        with open(path+img2img_js, 'r') as f:
            js = json.load(f)
        
        return js