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
    

    MONGODB_USERNAME:str
    MONGODB_PASSWORD:str
    MONGODB_URI:str
    FILE_MAX_SIZE:int
    FILE_DEFAULT_CHUNK_SIZE:int

    JWT_SECRET_KEY:str
    JWT_ALGORITHM:str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES:int

    GOOGLE_CLIENT_ID:str
    GOOGLE_CLIENT_SECRET:str
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }

def get_settings():
    return Settings()