import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


DATABASE_URL = "sqlite:///./moderation_logs.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLogs(Base):
    """Модель базы данных для логов предсказаний"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    toxic_model_res = Column(Boolean, nullable=False)
    spam_model_res = Column(Boolean, nullable=False)
    final_decision = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()