from tensorflow.keras.optimizers import Adam


def tuning(model):
    model.trainable = True

    for layer in model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
