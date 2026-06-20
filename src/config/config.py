from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    APP_NAME:Optional[str]="Nabatna-AI"
    APP_VERSION:Optional[str]="1.0.0"
    
    IMAGE_SIZE: Optional[int]=224
    NUM_CLASSES: Optional[int]=15
    BATCH_SIZE: Optional[int]=16
    EPOCHS: Optional[int]=10
    LEARNING_RATE: Optional[float]=0.0005

    TRAINED_MODEL_PATH:Optional[str]="models/model.h5"
    PROCESSED_DATA_PATH:Optional[str]="data/processed"
    RAW_DATA_PATH:Optional[str]="data/raw"
    CUSTOM_DATA_PATH:Optional[str]="data/custom"

    FILE_ALLOWED_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
    FILE_MAX_SIZE: Optional[int] = 1048576
    
    MONGODB_USERNAME: Optional[str] = None
    MONGODB_PASSWORD: Optional[str] = None
    MONGODB_URI: Optional[str] = None

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str 
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: Optional[int] = 60
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

def get_settings():
    return Settings()