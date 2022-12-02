
from server import http,websocket
from fastapi_profiler.profiler_middleware import PyInstrumentProfilerMiddleware

# app = http.app
import fastapi
app = fastapi.FastAPI()

### INIT CONFIG
import config
print(config.HttpConfig.HOST)

### INIT LOGGING
# from logging import debug


@app.on_event("startup")
async def on_startup():
    print("startup")
    from app import face_detection,face_recognition,super_resolution

    use_cuda = False
    face_detection.load_model(use_cuda)
    await super_resolution.load_model(use_cuda)
    face_recognition.load_model_adaface(use_cuda)

### FILE HANDLER
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from fastapi.requests import Request


### API HANDLER
import routers
from fastapi import APIRouter
api = APIRouter(prefix="/api")
# api.include_router(routers.v1)
api.include_router(routers.api)
app.include_router(api)

web = APIRouter(prefix="")
web.include_router(routers.web)
app.include_router(web)

from fastapi.staticfiles import StaticFiles
app.mount("/assets", StaticFiles(directory="public/assets"), name="assets")

### WEBSOCKET HANDLER
import server.websocket as websocket
app.mount("/", websocket.ws)


### MIDDLEWARE : CORS
from starlette.middleware.cors import CORSMiddleware
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"] )

# ### MIDDLEWARE : EXCEPTION HANDLER
# from exception import AppError,InternalServerError
# from fastapi.requests import Request
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError

# @app.exception_handler(AppError)
# async def InternalServerErrorHandler(request: Request, exception: AppError):
#     return JSONResponse(status_code = exception.status_code, content = exception.response())

# @app.exception_handler(Exception)
# async def InternalServerErrorHandler(request: Request, exception: Exception):
#     err = InternalServerError()
#     return JSONResponse(status_code = err.status_code, content = err.response())

# @app.exception_handler(RequestValidationError)
# async def RequestValidationErrorHandler(request: Request, exception: RequestValidationError):
#     loc = exception.errors()[0]["loc"]
#     msg = exception.errors()[0]["msg"]
#     return JSONResponse(status_code=422,content = {"status":0,"data":None,"error":{
#         "status":"RequestValidationError",
#         "message":"{} in '{}': {}".format(msg,loc[0],loc[1])
#     }})



### WEB SERVER
import uvicorn
if __name__ == '__main__':
    import config,os
    conf = config.HttpConfig()
    uvicorn.run(app='main:app', host=conf.HOST, port=conf.PORT, reload=True, debug=conf.DEBUG, reload_dirs = [
        os.getcwd()+"/app",
        os.getcwd()+"/app/*",
        os.getcwd()+"/infra",
        os.getcwd()+"/routers",
        os.getcwd()+"/server",
        os.getcwd()+"/utils",
        os.getcwd()+"/*.py",
    ])

