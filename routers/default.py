import base64
from fastapi import APIRouter, File, Form
router = APIRouter()
# from app.face_recognition import preprocess 
from app import face_recognition 


from app import face_detection
import uuid

@router.get("/")
async def home():
    """ Test Home """
    return {"msg": "home"}


@router.post("/detect")
async def detect_images(img: str = Form(...),date: str = Form(...)):
    """ Upload File """
    images = img
    print(date)

    id = date.replace(":","-").replace(".","-")
    results = {
        "base_img" : "assets/photo/{}/base.jpg".format(id),
        "date" : date,
        "status" : "complete",
        "data": []
    }
    list_images = face_detection.crop(id,images)
    for img in list_images:
        result = {
            "is_detected" : True,
            "name" : str(uuid.uuid4()),
            "nim" : str(uuid.uuid1())[0:8],
            "img" : img,
        }
        results["data"].append(result)

    return results
