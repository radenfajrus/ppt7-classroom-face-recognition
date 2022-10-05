from pydantic import BaseModel

class Error(BaseModel):
    code: str
    status: str
    message: str
class ApiError(BaseModel):
    status: int
    errors: Error

InternalServerError = type('InternalServerError',(ApiError,),{})


