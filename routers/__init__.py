from fastapi import APIRouter


from . import default
from . import files

latest = APIRouter()
latest.include_router(default.router)
latest.include_router(files.router)

v1 = APIRouter(prefix="/v1")
v1.include_router(default.router)
v1.include_router(files.router)

