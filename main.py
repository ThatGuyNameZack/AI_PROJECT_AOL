from flask import Flask, request, jsonify
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
logging.basicConfig(level=logging.DEBUG)


#were using flask for the app 
app = Flask("ImageClass")

# Define fuzzy variables for brightness and sharpness
brightness = ctrl.Antecedent(np.arange(0, 101, 1), 'brightness')
sharpness = ctrl.Antecedent(np.arange(0, 101, 1), 'sharpness')
image_class = ctrl.Consequent(np.arange(0, 101, 1), 'image_class')

#  Define membership functions for brightness and sharpness
brightness['low'] = fuzz.trimf(brightness.universe, [0, 0, 50])
brightness['medium'] = fuzz.trimf(brightness.universe, [25, 50, 75])
brightness['high'] = fuzz.trimf(brightness.universe, [50, 100, 100])

sharpness['blurry'] = fuzz.trimf(sharpness.universe, [0, 0, 50])
sharpness['medium'] = fuzz.trimf(sharpness.universe, [25, 50, 75])
sharpness['clear'] = fuzz.trimf(sharpness.universe, [50, 100, 100])

#  Define membership functions for image class (wanted or unwanted)
image_class['unwanted'] = fuzz.trimf(image_class.universe, [0, 0, 50])
image_class['neutral'] = fuzz.trimf(image_class.universe, [25, 50, 75])
image_class['wanted'] = fuzz.trimf(image_class.universe, [50, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(brightness['low'] & sharpness['blurry'], image_class['unwanted'])
rule2 = ctrl.Rule(brightness['medium'] & sharpness['medium'], image_class['neutral'])
rule3 = ctrl.Rule(brightness['high'] & sharpness['clear'], image_class['wanted'])

#  Create a control system
image_filter_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
image_filtering = ctrl.ControlSystemSimulation(image_filter_ctrl)

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.json
    brightness_value = data.get('brightness')
    sharpness_value = data.get('sharpness')

    if brightness_value is None or sharpness_value is None:
        return jsonify({"error": "Missing brightness or sharpness values"}), 400
    
    if not (0 <= brightness_value <= 100) or not (0 <= sharpness_value <= 100):
        return jsonify({"error": "Brightness and sharpness values must be between 0 and 100"}), 400

    # Simulate input values
    image_filtering.input['brightness'] = brightness_value
    image_filtering.input['sharpness'] = sharpness_value

    #  Compute the output
    image_filtering.compute()

    #  Show the result (classification of the image)
    classification = image_filtering.output['image_class']
    
    # Interpret the output (convert the score to a category)
    if classification < 33.3:
        result = "unwanted"
    elif classification < 66.6:
        result = "neutral"
    else:
        result = "wanted"

    return jsonify({
        "brightness": brightness_value,
        "sharpness": sharpness_value,
        "classification": result,
        "output_score": classification
    })

@app.route('/')
def home():
    app.logger.debug("home route accessed")
    return "Welcome to the Image Classification"

if __name__ == '__main__':
    app.run(debug=True)
