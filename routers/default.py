import base64
from io import BytesIO
from fastapi import APIRouter, File, Form
router = APIRouter()
# from app.face_recognition import preprocess 
from app import face_recognition, super_resolution 


from app import face_detection
import uuid
from PIL import Image

@router.get("/")
async def home():
    """ Test Home """
    return {"msg": "home"}


@router.post("/detect")
async def detect_images(img: str = Form(...),date: str = Form(...)):
    """ Upload File """
    img = img.split(",")[1] if "data:image" in img else img
    images = img
    print(date)

    id = date.replace(":","-").replace(".","-")
    results = {
        "base_img" : "assets/photo/{}/base-boxes.jpg".format(id),
        "date" : date,
        "status" : "complete",
        "data": []
    }

    data = {}

    list_res = face_detection.crop(id,images)
    # list_images_for_sr = []
    # print(list_res)
    # for res in list_res:
    #     idx = res.get("idx")
    #     param = {
    #         "idx": idx,
    #         "img" : res.get("img"),
    #     }
    #     list_images_for_sr.append(param)

    print("start super_resolution")
    list_res = await super_resolution.do_sr_multiple(id,list_res)
    list_images_for_fr = []
    for res in list_res:
        idx = res.get("idx")
        imgname = res.get("imgname")

        param = {
            "idx": idx,
            "imgname": imgname,
            "is_sr": res.get("is_sr"),
            "img_path" : res.get("img_sr_path"),
            "img" : "assets/photo{}/{}/{}".format( "_sr" if res.get("is_sr") else "" , id ,imgname),
        }
        data[idx] = param
        list_images_for_fr.append(param)
    
    print("start face_recognition")
    list_res = await face_recognition.predict(id,list_images_for_fr)
    for res in list_res:
        idx = res.get("idx")
        data[idx]["is_detected"] = res.get("is_detected")
        data[idx]["name"] = res.get("name")
        data[idx]["nim"] = res.get("nim")

    for i in data:
        results["data"].append(data[i])

    return results

import os,glob
@router.post("/photo")
async def detect_images(nim: str = Form(...),img_1: str = Form(...),img_2: str = Form(...),img_3: str = Form(...)):
    """ Upload File """
    if not os.path.isdir("public/assets/photo/{}".format(nim)):
        os.makedirs("public/assets/photo/{}".format(nim))

    files = glob.glob('public/assets/photo/{}/*'.format(nim))
    for f in files:
        os.remove(f)
    
    img_1 = img_1.split(",")[1] if "data:image" in img_1 else img_1
    img_2 = img_2.split(",")[1] if "data:image" in img_2 else img_2
    img_3 = img_3.split(",")[1] if "data:image" in img_3 else img_3

    f1 = Image.open(BytesIO(base64.b64decode(img_1)))
    f2 = Image.open(BytesIO(base64.b64decode(img_2)))
    f3 = Image.open(BytesIO(base64.b64decode(img_3)))
    f1.save("public/assets/photo/{}/0_l.jpg".format(nim))
    f2.save("public/assets/photo/{}/1_f.jpg".format(nim))
    f3.save("public/assets/photo/{}/2_r.jpg".format(nim))

    return [
        "/assets/photo/{}/0_l.jpg".format(nim),
        "/assets/photo/{}/1_f.jpg".format(nim),
        "/assets/photo/{}/2_r.jpg".format(nim),
    ]

@router.get("/photo/{nim}")
async def detect_images(nim: str):
    """ Get File """
    list_images = []
    if not os.path.isdir("public/assets/photo/{}".format(nim)):
        return list_images

    files = glob.glob('public/assets/photo/{}/*'.format(nim))
    buffered = BytesIO()
    for f in files:
        # image = Image.open(f)
        # image.save(buffered, format="JPEG")
        # img = base64.b64encode(buffered.getvalue())
        # list_images.append(img)

        filename = f.split("/")[-1].split("\\")[-1]
        img = "/assets/photo/{}/{}".format(nim,filename)
        list_images.append(img)

    return list_images
