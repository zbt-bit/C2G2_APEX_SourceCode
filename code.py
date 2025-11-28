import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier # Required for joblib loading, though not directly used below

# --- CONFIGURATION & SETUP ---

app = Flask(__name__)
CORS(app)  # <-- ADD THIS LINE right after creating the app
MODEL_FILE = 'street_lamp_model.joblib'
FEATURES = ['humidity', 'cloudcover', 'visibility', 'uvindex', 'day_of_year', 'temp', 'precip']

# --- LOAD THE TRAINED MODEL ---
try:
    # Model must be trained and saved by train_model.py first!
    model_clf = joblib.load(MODEL_FILE)
    print(f"Loaded AI Model from {MODEL_FILE}")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_FILE}' not found. Please run 'python train_model.py' first.")
    exit()

# --- AI PREDICTION FUNCTION (unchanged) ---
def predict_max_light_required(weather_data: dict) -> int:
    """Uses the trained model to predict whether Max Light (1) or Dimmer Light (0) is required."""
    input_df = pd.DataFrame([weather_data], columns=FEATURES)
    prediction = model_clf.predict(input_df)[0]
    return int(prediction)

# --- INTEGRATED CONTROL SYSTEM FUNCTION (unchanged) ---
def street_lamp_control_system(is_night_time: bool, 
                               is_motion_detected: bool, 
                               real_time_weather: dict) -> str:
    """Implements the hierarchical control logic: Time -> Weather -> Motion."""
    
    if not is_night_time:
        return "OFF (Daytime)"

    max_light_required = predict_max_light_required(real_time_weather)
    
    if max_light_required == 1:
        return "MAX OUTPUT (Safety Override: Rain/Fog)"

    if is_motion_detected:
        return "DIMMER OUTPUT (Motion Detected)"
    else:
        return "OFF (No Motion Detected)"

# --- FLASK API ENDPOINT ---
@app.route('/api/control_lamp', methods=['POST'])
def control_lamp_endpoint():
    """
    API endpoint that receives sensor and weather data and returns the lamp action.
    
    Expected JSON body:
    {
      "is_night_time": true, 
      "is_motion_detected": false,
      "weather_data": {
        "humidity": 75.0, 
        "cloudcover": 60.0,
        "visibility": 9.0, 
        "uvindex": 3,
        "day_of_year": 300, 
        "temp": 26.0, 
        "precip": 0.5
      }
    }
    """
    try:
        data = request.get_json(force=True)
        
        is_night = data['is_night_time']
        is_motion = data['is_motion_detected']
        weather = data['weather_data']
        
        # Call the core AI logic
        lamp_action = street_lamp_control_system(is_night, is_motion, weather)
        
        return jsonify({
            "status": "success",
            "lamp_action": lamp_action,
            "weather_prediction": "Max Light" if predict_max_light_required(weather) == 1 else "Dimmer/Off",
            "message": "Street lamp action determined successfully."
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }), 400

# --- RUN THE APP ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from other machines (like a frontend server)
    app.run(host='0.0.0.0', port=5000)