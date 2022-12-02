from concurrent.futures import as_completed
from datetime import datetime
import gc

import numpy as np

try:
    from realesrgan import RealESRGANer
except Exception as e:
    from .realesrgan import RealESRGANer

from basicsr.archs.rrdbnet_arch import RRDBNet
import torch

import cv2
import os,glob

upsampler = None

async def load_model(use_cuda=False):
    global upsampler 
    now = datetime.now()
    device = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"

    tile = 0 
    netscale = 4
    tile_pad = 10
    pre_pad = 0
    half = True if (torch.cuda.is_available() and use_cuda) else False

    dni_weight = None
    
    model_path = os.path.join(os.path.dirname(__file__),'model/RealESRGAN_x4plus.pth')
    model_dir = os.path.join(os.path.dirname(__file__),'model')
    if not os.path.exists(model_path):
        os.system(
            'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ' + model_dir
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half)


    img = np.zeros((112,112),dtype=np.uint8)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            img[j,i]=i

    #img =  cv2.imread("public/assets/photo_sr/2022-12-02T03-26-41-299Z/0.jpg",cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=1)

    _ = gc.collect()
    print("load model SR ({}) : {} ms".format(device,int((datetime.now() - now).total_seconds() * 1000)))

read_dir_template = "./public/assets/photo/{}".format
save_dir_template = "./public/assets/photo_sr/{}".format

def load_model_sync(use_cuda=False):
    now = datetime.now()
    device = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"

    tile = 0 
    netscale = 4
    tile_pad = 10
    pre_pad = 0
    half = True if (torch.cuda.is_available() and use_cuda) else False

    dni_weight = None
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = os.path.join(os.path.dirname(__file__),'model/RealESRGAN_x4plus.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half)
    
    print("load model SR ({}) : {} ms".format(device,int((datetime.now() - now).total_seconds() * 1000)))

    img_path = "public/assets/photo/2022-12-01T04-32-46-584Z/0.jpg"
    res = upsampling(0,"2022-12-01T04-32-46-584Z",img_path,upsampler)

    return upsampler

def upsampling(idx,id,img_path,upsampler):
    now = datetime.now()

    preprocessing_time = None
    face_enhancer_time = None
    upscaling_time = None
    outscale = None
    img_mode = None

    imgname = os.path.basename(img_path)
    imgnamepart, extension = os.path.splitext(imgname)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # img = np.asarray(img)
    h, w = img.shape[0:2]
    # h,w = img.size

    limit = 1
    if h < limit or w < limit:
        return {
            "idx" :  idx,
            "id" :  id,
            "is_sr" :  False,
            "img_non_sr_path" : img_path,
            "img_sr_path" : img_path,
            "imgname" :  imgname,
            "preprocessing_time" :  preprocessing_time,
            "face_enhancer_time" : face_enhancer_time,
            "upscaling_time" :  upscaling_time,
            "input_dir" :  read_dir_template(id),
            "output_dir" :  save_dir_template(id),
            "img_mode" :  img_mode,
            "extension" :  extension,
            "w" :  w,
            "h" :  h,
            "outscale" :  outscale,
            "o_width" :  w,
            "o_height" :  h,
        }

    batch = datetime.now()
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    elif len(img.shape) == 2:
        img_mode = None

    max_height = 112
    output_height = 112
    # if h < max_height:
    #     w = w/h * max_height
    #     h = max_height
    #     img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # w = int(w/h * max_height)
    # h = int(max_height)
    print(w, flush=True)
    print(h, flush=True)
    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # img = img.resize((w,h))

    preprocessing_time = int((datetime.now() - batch).total_seconds() * 1000)
    print("preprocessing time : {} ms".format(preprocessing_time), flush=True)

    try:
        batch = datetime.now()
        # outscale = 1
        outscale = output_height/h
        output, _ = upsampler.enhance(img, outscale=outscale)
        # output = img
        upscaling_time = int((datetime.now() - batch).total_seconds() * 1000)
        print("upsampler time : {} ms".format(upscaling_time), flush=True)
    except RuntimeError as error:
        print('Error', error, flush=True)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.', flush=True)

    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = '.png'

    o_height, o_width = output.shape[0:2]
    # o_height, o_width = output.size
    # o_width = int(w/h * 512)
    # o_height = int(512)
    # output = cv2.resize(output, (o_width, o_height), interpolation=cv2.INTER_LANCZOS4)
    
    imgname = f'{imgnamepart}{extension}'
    save_path = save_dir_template(id) +"/"+ imgname
    # output.save(save_path)
    cv2.imwrite(save_path,output)

    print("upsampling SR : {} ms".format(int((datetime.now() - now).total_seconds() * 1000)), flush=True)

    return {
        "idx" :  idx,
        "id" :  id,
        "is_sr" :  True,
        "img_non_sr_path" : img_path,
        "img_sr_path" : save_path,
        "imgname" :  imgname,
        "preprocessing_time" :  preprocessing_time,
        "face_enhancer_time" : face_enhancer_time,
        "upscaling_time" :  upscaling_time,
        "input_dir" :  read_dir_template(id),
        "output_dir" :  save_dir_template(id),
        "img_mode" :  img_mode,
        "extension" :  extension,
        "w" :  w,
        "h" :  h,
        "outscale" :  outscale,
        "o_width" :  o_width,
        "o_height" :  o_height,
    }

# def upsampling_mp(idx,id,img,imgname):
#     return await upsampling(idx,id,img,imgname)
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
async def run_in_process(executor, fn, *args):
    loop = asyncio.get_event_loop()
    # try:
    #     executor = app.state.executor
    # except Exception as e:
    #     executor = ProcessPoolExecutor(4)
    # with ProcessPoolExecutor() as executor:
    res = await loop.run_in_executor(executor, fn, *args)  # wait and return result
    return res

import pandas as pd
async def do_sr(id):
    start = datetime.now()

    read_dir = read_dir_template(id)
    save_dir = save_dir_template(id)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    data_monitor = []
    tasks = []
    results = []

    executor = ProcessPoolExecutor()
    paths = sorted(glob.glob(os.path.join(read_dir, '*')))
    for idx, path in enumerate(paths[0:4]):
        imgname, extension = os.path.splitext(os.path.basename(path))

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_input = img.copy()

        tasks.append(run_in_process(executor,upsampling,idx,id,img_input,imgname,upsampler))
            
    print("run taskgroup")
    # python app/super_resolution/__init__.py
    
    results = await asyncio.gather(*tasks)
    executor.shutdown() 
    for res in results:
        if res is not None:
            data_monitor.append({
                "idx" :   res.get("idx"),
                "id" :   res.get("id"),
                "imgname" :   res.get("imgname"),
                "preprocessing_time" :  res.get("preprocessing_time"),
                "face_enhancer_time" : res.get("face_enhancer_time"),
                "upscaling_time" :  res.get("upscaling_time"),
                "input_path" :  res.get("input_path"),
                "output_path" :  res.get("save_path"),
                "img_mode" :  res.get("img_mode"),
                "extension" :  res.get("extension"),
                "width" :  res.get("w"),
                "height" :  res.get("h"),
                "outscale" :  res.get("outscale"),
                "o_width" :  res.get("o_width"),
                "o_height" :  res.get("o_height"),
            })
            results.append({
                "idx": res.get("idx"),
                "is_sr": res.get("is_sr"),
                "img_non_sr" : res.get("img_sr"),
                "img_sr" : res.get("img"),
            })


    print("Finish all data {} : {} ms".format(len(paths),int((datetime.now() - start).total_seconds() * 1000)))
    df_monitor = pd.DataFrame(data_monitor)
    print(df_monitor)
    df_monitor.to_csv("sr_{}.csv".format(id),index=False,header=True)

    return results

def upsampling2(idx,id,img_path):
    return upsampling(idx,id,img_path,upsampler)
async def upsampling3(idx,id,img_path):
    return upsampling(idx,id,img_path,upsampler)
async def do_sr_multiple(id,list_data):
    start = datetime.now()

    read_dir = read_dir_template(id)
    save_dir = save_dir_template(id)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    data_monitor = []
    tasks = []
    tasks2 = []
    results = []

    executor = ProcessPoolExecutor(8)
    # for data in [list_data[0],list_data[1]]:
    sr_results1 = []
    for data in list_data:
        idx = data.get("idx")
        img_path = data.get("img_path")
        
        sr_results1.append(upsampling2(idx,id,img_path))
        
        _ = gc.collect()
        # tasks.append(executor.submit(upsampling, idx,id,img_path,upsampler))
        # tasks2.append(upsampling3(idx,id,img_path))
            
    print("run taskgroup")
    # python app/super_resolution/__init__.py

    sr_results = []
    try:
        # results = await asyncio.gather(*tasks)
        for future in as_completed(tasks):
            # get the result for the next completed task
            sr_results.append(future.result()) # blocks
    finally:
        executor.shutdown()

    results3 = await asyncio.gather(*tasks2)

    for res in sr_results1:
        if res is not None:
            data_monitor.append({
                "idx" :   res.get("idx"),
                # "id" :   res.get("id"),
                # "imgname" :   res.get("imgname"),
                # "preprocessing_time" :  res.get("preprocessing_time"),
                # "face_enhancer_time" : res.get("face_enhancer_time"),
                "upscaling_time" :  res.get("upscaling_time"),
                # "input_path" :  res.get("input_path"),
                # "output_path" :  res.get("save_path"),
                # "img_mode" :  res.get("img_mode"),
                # "extension" :  res.get("extension"),
                "width" :  res.get("w"),
                "height" :  res.get("h"),
                "outscale" :  res.get("outscale"),
                "o_width" :  res.get("o_width"),
                "o_height" :  res.get("o_height"),
            })
            results.append({
                "idx": res.get("idx"),
                "imgname" :   res.get("imgname"),
                "is_sr": res.get("is_sr"),
                "img_non_sr_path" : res.get("img_non_sr_path"),
                "img_sr_path" : res.get("img_sr_path"),
            })


    print("Finish all data {} : {} ms".format(len(list_data),int((datetime.now() - start).total_seconds() * 1000)))
    df_monitor = pd.DataFrame(data_monitor)
    print(df_monitor)
    df_monitor.to_csv("logs/sr_{}.csv".format(id),index=False,header=True)

    return results

async def main():
    id = "2022-11-07T09-20-46-529Z"
    await load_model()
    print(upsampler)
    do_sr(id)

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    
    img_path = "public/assets/photo/2022-12-01T04-32-46-584Z/0.jpg"
    upsampler = load_model_sync()
    # upsampling
    res = upsampling(0,"2022-12-01T04-32-46-584Z",img_path,upsampler)
    print(res)
