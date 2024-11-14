import os
import keras
from keras.models import load_model

base_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_dir, 'emotion_model.h5')
model = load_model(model_path)

def predict_emotion(input_data):
    prediction = model.predict(input_data)
    return prediction