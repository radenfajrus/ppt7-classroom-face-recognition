from fastapi import APIRouter
router = APIRouter()
# from app.face_recognition import preprocess 
# from app.face_recognition import detect 

# @router.get("/")
# async def home():
#     img = req.body

#     img_clean = preprocess(img)
#     results = detect(img_clean)

#     """ Test Home """
#     return {"data": results}

@router.get("/aaa")
async def home():
    """ Test Home """
    return {"msg": "homeaaa"}
