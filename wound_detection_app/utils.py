import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model("model/wound_model.h5")

medical_advice = {
    'Abrasions': {'urgent': False, 'home_care': True, 'advice': 'Clean gently, apply antiseptic, bandage'},
    'Bruises': {'urgent': False, 'home_care': True, 'advice': 'Apply ice, rest, monitor for changes'},
    'Burns': {'urgent': True, 'home_care': False, 'advice': 'Cool with water, seek medical help immediately'},
    'Cut': {'urgent': False, 'home_care': True, 'advice': 'Clean, apply pressure, bandage. See doctor if deep'},
    'Diabetic Wounds': {'urgent': True, 'home_care': False, 'advice': 'See doctor immediately'},
    'Laceration': {'urgent': True, 'home_care': False, 'advice': 'Apply pressure, go to emergency room'},
    'Normal': {'urgent': False, 'home_care': True, 'advice': 'Healthy skin, no treatment needed'},
    'Pressure Wounds': {'urgent': True, 'home_care': False, 'advice': 'Need professional medical care'},
    'Surgical Wounds': {'urgent': False, 'home_care': True, 'advice': 'Follow doctor instructions, keep clean'},
    'Venous Wounds': {'urgent': True, 'home_care': False, 'advice': 'Need professional medical care'}
}

def analyze_wound(image_path):
    img = keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    
    class_names = list(model.class_names) if hasattr(model, 'class_names') else list(medical_advice.keys())
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    advice = medical_advice.get(predicted_class, {})

    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'urgent': advice.get('urgent'),
        'home_care': advice.get('home_care'),
        'advice': advice.get('advice')
    }

    return predicted_class, confidence, result
