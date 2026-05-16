from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from sqlalchemy.orm import Session

from database import get_db
from schemas import PredictionResponse, PredictionRequest
from service import PredictionService
from model_manager import model_manager
from config import settings


router = APIRouter(prefix="/api", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
@settings.limiter.limit("3000/minute")
async def prediction(
    request:Request,
    prediction_request:PredictionRequest,
    db:Session = Depends(get_db)
) -> PredictionResponse: 
    "Основной API для предсказания нарушений"
    try:
        response = PredictionService.predict_content(request=prediction_request)
        PredictionService.save_prediction_to_db(db=db, request=prediction_request, response=response)
        
        return response
    
    except Exception as e:
        if db is not None:
            db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/health")
async def health_check() -> dict:
    "Проверка: загружены ли моедли"
    return {
        "is_toxic_model_loaded": model_manager.toxic_model is not None,
        "is_spam_model_loaded": model_manager.spam_model is not None
    }