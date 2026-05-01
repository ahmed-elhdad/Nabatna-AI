

from .config import get_settings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
settings=get_settings()
def load_data():

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        settings.PROCESSED_DATA_PATH,
        target_size=(settings.IMAGE_SIZE,settings.IMAGE_SIZE),
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        settings.PROCESSED_DATA_PATH,
        target_size=(settings.IMAGE_SIZE,settings.IMAGE_SIZE),
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )