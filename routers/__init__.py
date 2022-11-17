from fastapi import APIRouter


from . import default
from . import files
from . import home_web
from . import mobile_web

api = APIRouter()
api.include_router(default.router)
api.include_router(files.router)

# v1 = APIRouter(prefix="/v1")
# v1.include_router(default.router)
# v1.include_router(files.router)

web = APIRouter()
web.include_router(home_web.router)
web.include_router(mobile_web.router)