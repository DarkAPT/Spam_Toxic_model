from pydantic_settings import BaseSettings, SettingsConfigDict
from slowapi import Limiter
from slowapi.util import get_remote_address

class Settings(BaseSettings):
    "Класс настроик приложения"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    limiter:Limiter = Limiter(key_func=get_remote_address)
    
    #Настройки базы данных(PostgreSQL)
    postgres_host: str = "localhost"
    postgres_port: str = "5432"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_db_name: str = "moderation_system_db"
    
    @property
    def database_url(self) ->str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db_name}"
    
    # Настройки моделей
    toxic_model_path: str = "../models/toxic"
    spam_model_path: str = "../models/spam/spam_model2.pth"
    
    # Настройки порогов
    toxic_block_threshold: float = 0.95
    spam_block_threshold: float = 0.5
    toxic_review_threshold: float = 0.4
    spam_review_threshold: float = 0.3
    
    # Настройки сервера
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
settings = Settings()