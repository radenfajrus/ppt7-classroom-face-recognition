from typing import Dict

class Panic(Exception):
    """Base class for exceptions in this module."""


class Error(Exception):
    """Base class for exceptions in this module."""
    

class ServerPanic(Panic):
    lang: str = "en"
    message: Dict[str, str] = {
        "in": "Tidak dapat menjalankan server, tipe server belum di set -> Class (Http|gRPC|WebSocket)Server",
        "en": "Cannot start server, server type not set -> Class (Http|gRPC|WebSocket)Server",
    }
    def __str__(self):
        return self.message[self.lang]



class AppError(Error):
    status_code: str
    status: str
    dict_msg: Dict[str,str]
    lang: str
    def __init__(self, msg: str = None) -> None:
        self.custom_message = msg
    def __str__(self):
        return self.get_message()
    def get_message(self):
        return (self.custom_message) if(self.custom_message) else self.dict_msg[self.lang]
    def response(self):
        return {"status":0,"data":None,"error":{"status":self.status,"message":self.get_message()}}


class InternalServerError(AppError):
    status_code: str = 500
    status: str = "InternalServerError"
    dict_msg: Dict[str,str] = {
        "en":"Unknown Error, please contact Administrator",
        "in":"Terjadi kesalahan system, harap hubungi Admin"
    }
    lang: str = "en"




class InternalServerError(AppError):
    status_code: str = 500
    status: str = "InternalServerError"
    dict_msg: Dict[str,str] = {
        "en":"Unknown Error, please contact Administrator",
        "in":"Terjadi kesalahan system, harap hubungi Admin"
    }
    lang: str = "en"


