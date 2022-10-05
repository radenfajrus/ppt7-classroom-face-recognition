from fastapi import APIRouter
router = APIRouter(prefix="/files")

@router.get("/")
async def get_list_file():
    """ Get File Name"""
    return {"msg": "file"}

@router.post("/")
async def upload_file():
    """ Upload File """
    return {"msg": "file"}

@router.get("/{file_id}")
async def get_file_by_id():
    """ Get File By Id"""
    return {"msg": "file"}
