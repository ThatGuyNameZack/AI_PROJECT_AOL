from tensorflow.keras.models import load_model

# Path to your model
model_path = '/Users/rafisatria/Documents/GitHub/AI_PROJECT_AOL/emotion_model.h5'

try:
    # Load the model
    emotion_model = load_model(model_path)
    
    # Print the model summary
    emotion_model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
