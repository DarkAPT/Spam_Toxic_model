from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from database import Base, SessionLocal, engine
from auth import initial_moderator
from model_manager import model_manager
from routers import auth, prediction, user_stats,moderation

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        
        with SessionLocal() as db:
            initial_moderator(db)
        
        model_manager.load_spam_model()
        model_manager.load_toxic_model()
    
    except Exception as e:
        print("Ошибка при инициализации:{e}")
        raise
    
    yield

app = FastAPI(
    title="Система модерации контента",
    description="Система автоматической модерации",
    version="1.0.0",
    lifespan=lifespan
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Подключаем маршрутизаторы
app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(user_stats.router)
app.include_router(moderation.router)

# Статические роуты
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/moderator")
async def serve_moderator():
    return FileResponse("static/moderator.html")

@app.get("/stats")
async def serve_stats():
    return FileResponse("static/stats.html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
