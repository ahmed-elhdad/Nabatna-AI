def predict(model):
    predicted_labels=model.predict('../../data/plants/PlantVillage/Potato___Early_blight/00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early.B 8722.jpg')
    return predicted_labels
import joblib
model=joblib.load('saved_models/model.h5')
print(predict(model=model))