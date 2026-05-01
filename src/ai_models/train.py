from src.utils.data_loader import load_data
from src.helpers.config import get_settings
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
gpus = tf.config.list_physical_devices("GPU")
tf.debugging.set_log_device_placement(True)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Enabled 🚀")
    except:
        print("error to enable GPU")
with tf.device("/GPU:0"):



    settings=get_settings()
    train_data, val_data = load_data(settings.PROCESSED_DATA_PATH)
    # =========================
    # MODEL
    # =========================
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(settings.IMAGE_SIZE,settings.IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(train_data.class_names), activation="softmax"),
        ]
    )

    # =========================
    # COMPILE
    # =========================
    model.compile(
        optimizer=Adam(learning_rate=settings.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # =========================
    # CALLBACKS
    # =========================
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2),
    ]

    # =========================
    # TRAINING
    # =========================
    print("Training")
    history = model.fit(
        train_data, validation_data=val_data, epochs=settings.EPOCHS, callbacks=callbacks
    )
    print("Tuning")
    from . import tuning

    tuning.tuning(base_model)
    print("Dumping")
    import joblib

    joblib.dump(model, settings.TRAINED_MODEL_PATH)
