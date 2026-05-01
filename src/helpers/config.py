from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    WEATHER_API_KEY: str

    IMAGE_SIZE: int
    NUM_CLASSES: int
    BATCH_SIZE: int
    EPOCHS: int
    LEARNING_RATE: float

    TRAINED_MODEL_PATH:str="models/model.h5"
    PROCESSED_DATA_PATH:str="data/processed"
    RAW_DATA_PATH:str="data/raw"
    CUSTOM_DATA_PATH:str="data/custom"

    model_config = SettingsConfigDict(env_file='.env')


def get_settings():
    return Settings()