from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI,Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uvicorn
from models import PredictionLogs, get_db
from model_manager import model_manager

app = FastAPI(
    title="Система модерации контента",
    description="Система автоматической модерации",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str
    
class PredictionResponse(BaseModel):
    decision:str
    spam_confidence:float
    toxic_confidence:float
    
    class Config:
        from_attributes=True

class PredictionLogsItem(BaseModel):
    id: int
    text: str
    timestamp: datetime
    toxic_model_res: bool
    spam_model_res: bool
    final_decision: str
    confidence: float
    
    class Config:
        from_attributes = True
    
@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_toxic_model()
        model_manager.load_spam_model()
        print("Модели загружены успешно")
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")

@app.get("/health")
async def health_check():
    return {
        "toxic_model_loaded": model_manager.toxic_model is not None,
        "spam_model_loaded": model_manager.spam_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, db: Session=Depends(get_db)) -> PredictionResponse:
    """
    Predict endpoint для классификации текста

    Анализирует текст на токсичность и спам, принимает решение
    (block/review/allow) на основе пороговых значений моделей

    Сохраняет результаты в базу данных и возвращает решение
    с уверенностью каждой модели
    """
    try:
        toxic_probability = model_manager.toxic_predict(request.text)
        spam_probability = model_manager.spam_predict(request.text)
        
        toxic_prediction = 1 if toxic_probability > 0.95 else 0
        spam_prediction = 1 if spam_probability > 0.5 else 0
        
        if toxic_prediction or spam_prediction:
            decision = "block"
        elif toxic_probability > 0.4 or spam_probability > 0.3:
            decision = "review"
        else:
            decision = "allow"
        
        log_entry = PredictionLogs(
            text = request.text[:1000],
            toxic_model_res = bool(toxic_prediction),
            spam_model_res = bool(spam_prediction),
            final_decision = decision,
            confidence = round(max(toxic_probability,spam_probability),3)
        )
        
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        return PredictionResponse(
            decision=decision,
            spam_confidence=round(spam_probability,3),
            toxic_confidence=round(toxic_probability,3)
        )
        
    except Exception as e:
        if 'db' in locals:
            db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/get_prediction_logs", response_model=List[PredictionLogsItem])
async def get_prediction_logs(
    db: Session = Depends(get_db),
    decision: Optional[str] = Query(None, description="Фильтр по решению(block, review, allow)"),
    limit: int = Query(10, ge=1 ,le=100),
    offset: int = Query(0, ge=0)
):
    """
    Эндпоинт для получения истории предсказаний из базы данных

    Параметры запроса (query parameters):
        - decision: фильтрация по решению модерации (block/review/allow)
        - limit: количество записей на странице (от 1 до 100, по умолчанию 10)
        - offset: смещение для пагинации (по умолчанию 0)

    Возвращает список записей, отсортированных по времени (сначала новые)
    """
    try:
        query = db.query(PredictionLogs)
        
        if decision:
            query = query.filter(PredictionLogs.final_decision == decision)
        
        query = query.order_by(PredictionLogs.timestamp.desc())
        items = query.offset(offset).limit(limit).all()
        
        return items
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(
        "app:main",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
