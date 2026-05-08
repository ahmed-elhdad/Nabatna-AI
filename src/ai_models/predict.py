from fastapi import UploadFile
import tensorflow as tf
from src.helpers.config import get_settings,Settings
settings=Settings()
model=tf.keras.models.load_model(settings.TRAINED_MODEL_PATH)
def predict(img:UploadFile):
    predicted_labels=model.predict(img)
    return predicted_labels
