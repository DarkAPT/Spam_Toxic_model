import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings


engine = create_engine(url=settings.database_url)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLogs(Base):
    """Модель базы данных для логов предсказаний"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    toxic_model_res = Column(Boolean, nullable=False)
    spam_model_res = Column(Boolean, nullable=False)
    final_decision = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    moderator_conf = Column(Boolean, default=False)
    correct_label = Column(Boolean, default=True)
    
    moderation_actions = relationship("ModerationActions", back_populates="prediction_log")

class SystemUsers(Base):
    """Пользователи системы модерации"""
    __tablename__ = "system_users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    moderation_actions = relationship("ModerationActions", back_populates="moderator")
    
class ModerationActions(Base):
    """Логи действий модераторов"""
    __tablename__ = "moderation_actions"
    
    id = Column(Integer, primary_key=True)
    log_id = Column(Integer, ForeignKey('prediction_logs.id'), nullable=False, index=True)
    moderator_id = Column(Integer, ForeignKey('system_users.id'), nullable=False, index=True)
    action = Column(String(50), nullable=False)  # 'confirm', 'block', 'allow'
    previous_decision = Column(String(50))
    new_decision = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    prediction_log = relationship("PredictionLogs", back_populates="moderation_actions")
    moderator = relationship("SystemUsers", back_populates="moderation_actions")
    
class UserStats(Base):
    """Статистика пользователей"""
    __tablename__ = "user_stats"
    
    user_id = Column(String(255), primary_key=True)
    violations_count = Column(Integer, default=0)
    total_content = Column(Integer, default=0)
    last_violation = Column(DateTime, nullable=True)
    status = Column(String(50), default='active')  # 'active', 'warned', 'banned'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    "Инициализация базы данных (создание таблиц)"
    Base.metadata.create_all(bind=engine)
