from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/student_db"
)
engine = create_engine(DATABASE_URL,
    connect_args={"sslmode": "require"})

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Student(Base):

    __tablename__ = "student"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    study_hours = Column(Float)
    attendance = Column(Float)
    prev_mark = Column(Float)
    predicted = Column(Float)
    actual = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base.metadata.create_all(bind=engine)
