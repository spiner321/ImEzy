from typing import Union, Optional
from pydantic import BaseModel, Field


username = 'testname'

class txt2img_inputs(BaseModel):

    prompt: str = ''
    negative_prompt: str = ''
    prompt_style: str = '' # 'oil painting'
    prompt_style2: str = '' # 'space'
    steps: Optional[int] = Field(20, gt=0, lt=151) # 1 ~ 150
    sampler_index: Optional[int] = Field(0, min_digit=0, max_digit=3)
    restore_faces: Optional[bool] = Field(False)
    tiling: Optional[bool] = Field(False)
    n_iter: Optional[int] = Field(1, min_digit=1, max_digit=5)
    batch_size: Optional[int] = Field(4, min_digit=1, max_digit=8)
    cfg_scale: Optional[float] = Field(7.0, min_digit=0.1, max_digit=30.0)
    denoising_strength: Optional[float] = Field(0.75, min_digit=0.0, max_digit=1.0)
    seed: Optional[int] = -1
    subseed: Optional[int] = Field(-1, min_digit=-1, max_digit=1000)
    subseed_strength: Optional[float] = Field(0.0, min_digit=0.0, max_digit=1.0)
    seed_resize_from_h: Optional[int] = Field(0, min_digit=0, max_digit=2048)
    seed_resize_from_w: Optional[int] = Field(0, min_digit=0, max_digit=2048)
    seed_enable_extras: bool = True
    height: int = Field(512, min_digit=1, max_digit=2048)
    width: int = Field(512, min_digit=1, max_digit=2048)
    enable_hr: Optional[bool] = Field(False)
    scale_latent: Optional[bool] = Field(True)
    
    
class img2img_inputs(BaseModel):

    mode: int = 0
    prompt: str = ''
    negative_prompt: str = ''
    prompt_style: str = ''
    prompt_style2: str = ''
    init_img: str = ''
    init_img_with_mask: str = ''
    init_img_inpaint: str = ''
    init_mask_inpaint: str = ''
    mask_mode: str = ''
    steps: Optional[int] = Field(20, gt=0, lt=151) # 1 ~ 150
    sampler_index: Optional[int] = Field(0, min_digit=0, max_digit=3)
    mask_blur: int = 1
    inpainting_fill: int = 0
    restore_faces: Optional[bool] = Field(False)
    tiling: Optional[bool] = Field(False)
    n_iter: Optional[int] = Field(1, min_digit=1, max_digit=5)
    batch_size: Optional[int] = Field(4, min_digit=1, max_digit=8)
    cfg_scale: Optional[float] = Field(7.0, min_digit=0.1, max_digit=30.0)
    denoising_strength: Optional[float] = Field(0.75, min_digit=0.0, max_digit=1.0)
    seed: Optional[int] = -1
    subseed: Optional[int] = Field(-1, min_digit=-1, max_digit=1000)
    subseed_strength: Optional[float] = Field(0.0, min_digit=0.0, max_digit=1.0)
    seed_resize_from_h: Optional[int] = Field(0, min_digit=0, max_digit=2048)
    seed_resize_from_w: Optional[int] = Field(0, min_digit=0, max_digit=2048)
    seed_enable_extras: bool = True
    height: int = Field(512, min_digit=1, max_digit=2048)
    width: int = Field(512, min_digit=1, max_digit=2048)
    resize_mode: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    img2img_batch_input_dir: str = ''
    img2img_batch_output_dir: str = ''