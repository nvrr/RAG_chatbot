from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(
DATABASE_URL,
echo=False,
pool_pre_ping=True,
pool_recycle=1800
)


SessionLocal = sessionmaker(
    bind=engine,
    # autoFlash=False,
    autoflush=False,
    autocommit=False
)
