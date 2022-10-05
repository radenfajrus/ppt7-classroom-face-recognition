import config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

conf = config.DbConfig()


engine = create_engine(conf.DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
print("SessionLocal called")

from typing import Generator
async def get_client() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

