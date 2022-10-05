from fastapi import APIRouter
router = APIRouter()

@router.get("/")
def home():
    """ Test Home """
    return {"msg": "home"}
