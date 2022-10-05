
### INIT APP
import fastapi
app = fastapi.FastAPI()


print("twice") # This will run twice
@app.on_event("startup")
async def startup_event() -> None:
    print("once")  # This will run once

@app.on_event("shutdown")
async def on_shutdown() -> None:
    print("on_shutdown")

### INIT CONFIG
import config
print(config.HttpConfig.HOST)

### INIT LOGGING
from logging import debug


### FILE HANDLER
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from fastapi.requests import Request

templates = Jinja2Templates(directory="public")
pages = APIRouter()

@pages.get("/")
async def home(request: Request):
	return templates.TemplateResponse("index.html",{"request":request})
	
app.include_router(pages)


### API HANDLER
import routers
from fastapi import APIRouter
api = APIRouter(prefix="/api")
api.include_router(routers.v1)
api.include_router(routers.latest)
app.include_router(api)


### MIDDLEWARE : CORS
from starlette.middleware.cors import CORSMiddleware
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"] )

### MIDDLEWARE : EXCEPTION HANDLER
from exception import AppError,InternalServerError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(AppError)
async def InternalServerErrorHandler(request: Request, exception: AppError):
    return JSONResponse(status_code = exception.status_code, content = exception.response())

@app.exception_handler(Exception)
async def InternalServerErrorHandler(request: Request, exception: Exception):
    err = InternalServerError()
    return JSONResponse(status_code = err.status_code, content = err.response())

@app.exception_handler(RequestValidationError)
async def RequestValidationErrorHandler(request: Request, exception: RequestValidationError):
    loc = exception.errors()[0]["loc"]
    msg = exception.errors()[0]["msg"]
    return JSONResponse(status_code=422,content = {"status":0,"data":None,"error":{
        "status":"RequestValidationError",
        "message":"{} in '{}': {}".format(msg,loc[0],loc[1])
    }})



### WEB SERVER
import uvicorn
if __name__ == '__main__':
    import config
    http = config.HttpConfig()
    uvicorn.run(app='main:app', host=http.HOST, port=http.PORT, reload=True, debug=http.DEBUG)

