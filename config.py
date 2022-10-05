

from abc import ABC

from starlette.config import Config
from starlette.datastructures import Secret

config = Config(".env")

class Config(ABC):
    pass

class HttpConfig(Config):
    DEBUG = config("APP_DEBUG", cast=bool, default=False)
    THREADED = config("APP_THREADED", cast=bool, default=False)
    HOST = config("APP_HOST", cast=str, default='0.0.0.0')
    PORT = config("APP_PORT", cast=int, default='8000')
    SECRET_KEY = config("SECRET_KEY", cast=Secret, default="CHANGEME")


# class DbConfig(Config):
#     def __init__(self):
#         USER     = config("POSTGRES_USER", cast=str, default="")
#         PASSWORD = config("POSTGRES_PASSWORD", cast=Secret, default="")
#         HOST     = config("POSTGRES_SERVER", cast=str, default="127.0.0.1")
#         PORT     = config("POSTGRES_PORT", cast=str, default="5432")
#         NAME     = config("POSTGRES_DB", cast=str, default="")
#         DB_URL   = config("DATABASE_URL",cast=str,default=f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{NAME}")
#         super().__init__(DB_URL=DB_URL)

