import joblib
import os

# Load model lazily on first use
_model = None

def _load_model():
    """Load the model from disk (only once)"""
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models/model.h5')
        _model = joblib.load(model_path)
    return _model

def predict(file):
    model = _load_model()
    predicted_labels = model.predict(file)
    return predicted_labels