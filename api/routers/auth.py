from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session
from database import get_db, SystemUsers
from auth import auth_service
from schemas import UserRequest
from config import settings

router = APIRouter(prefix="/api/auth", tags=["authentication"])

@router.post("/login")
@settings.limiter.limit("5/minute")
async def login(
    request:Request,
    user_requset: UserRequest,
    db:Session = Depends(get_db)
):
    user = auth_service.auth_user(db=db,
                                username=user_requset.username,
                                password=user_requset.password)
    
    if not user:
        raise HTTPException(status_code=401,detail="Невернные данные")
    
    return {
        "message": "Успешный вход",
        "user_id": user.id,
        "username": user.username
    }

@router.post("/register")
@settings.limiter.limit("5/minute")
async def register(
    request:Request,
    user_request: UserRequest,
    db:Session = Depends(get_db),
    is_authenticated:SystemUsers = Depends(auth_service.get_current_moderator)
):
    try:
        user = auth_service.register_moderator(db=db,credentials=user_request)
        return {
            "message":"Модератор создан",
            "username": user.username,
            "id": user.id
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Ошибка при создании модератора")