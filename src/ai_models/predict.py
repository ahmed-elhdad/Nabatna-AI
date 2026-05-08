import tensorflow as tf
from src.helpers.config import Settings,get_settings
settings=Settings()
def predict(img):
    predicted_labels=tf.keras.models.load_model(settings.TRAINED_MODEL_PATH)
    return predicted_labels
