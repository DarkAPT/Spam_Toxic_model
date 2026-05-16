from passlib.context import CryptContext
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, status
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session

from schemas import UserRequest
from database import SystemUsers, get_db



pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"]
)
security = HTTPBasic()

class AuthService:
    @staticmethod
    def get_hashed_password(password):
        "Возвращает хэш пароля"
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(password, hashed_password):
        "Проверка пароля"
        return pwd_context.verify(password,hashed_password)
    
    @staticmethod
    def auth_user(
        db: Session,
        username: str,
        password: str
    ):
        user = db.query(SystemUsers).filter(SystemUsers.username == username).first()
        
        if user and AuthService.verify_password(
            password=password,
            hashed_password=user.hashed_password):
            return user
        else:
            return None
    
    @staticmethod
    def get_current_moderator(
        credentials: HTTPBasicCredentials = Depends(security),
        db: Session = Depends(get_db)
        ):
        
        user = AuthService.auth_user(db=db, username=credentials.username, password=credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Невернные учетный данные",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        return user
    
    @staticmethod
    def register_moderator(
        db:Session,
        credentials:UserRequest
    ):
        user = db.query(SystemUsers).filter(SystemUsers.username == credentials.username).first()
        
        if user:
            raise ValueError("Пользователь с таким именем уже сущетвует")
        
        new_user = SystemUsers(
            username = credentials.username,
            hashed_password = AuthService.get_hashed_password(credentials.password)
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user
        
    
def initial_moderator(db: Session):
    moderator = db.query(SystemUsers).filter(SystemUsers.username == "moderator").first()
    
    if not moderator:
        moderator = SystemUsers(
            username = "moderator",
            hashed_password = AuthService.get_hashed_password("moderator123")
        )
        
        db.add(moderator)
        db.commit()
        print("Создан начальный модератор: moderator / moderator123")

auth_service = AuthService()