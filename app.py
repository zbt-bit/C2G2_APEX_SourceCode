import pandas as pd
import joblib
import requests
import random
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION (WeatherAPI + Self-Diagnosis Failsafe) ---
app = Flask(__name__)
CORS(app) 
MODEL_FILE = 'street_lamp_model.joblib'

# CRITICAL FIX: This list MUST match the 7 features and EXACT order used in train_model.py.
FEATURES = [
    'humidity', 'cloudcover', 'visibility', 'uvindex', 'day_of_year', 
    'temp', 'precip' 
]

# Self-Diagnosis Failsafe Threshold
MAX_SAFE_TEMP_C = 55.0 

# Weather API Setup
WEATHER_API_KEY = '10012805f62e4da08c702836252911' 
LOCATION = 'Kuala Lumpur'
WEATHER_URL = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={LOCATION}&days=1&aqi=no&alerts=no"


def read_health_sensor():
    """
    Simulates reading the internal temperature sensor attached to the Edge device.
    """
    current_temp = random.uniform(30.0, 65.0) 
    
    is_overheated = current_temp > MAX_SAFE_TEMP_C
    
    return {
        'current_temp_c': round(current_temp, 1),
        'max_safe_temp_c': MAX_SAFE_TEMP_C,
        'is_overheated': is_overheated
    }


# Load the ML model 
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found. Please run train_model.py first.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- HELPER FUNCTIONS ---

def calculate_day_of_year(dt):
    """Calculates the day of the year (1-366)."""
    return dt.timetuple().tm_yday

def is_currently_night(sunrise_time_str, sunset_time_str):
    """Determines if the current time is between sunset and sunrise."""
    today = datetime.now()
    
    try:
        # WeatherAPI uses I:M p format (e.g., 07:15 AM)
        sunrise_dt = datetime.strptime(f"{today.strftime('%Y-%m-%d')} {sunrise_time_str}", '%Y-%m-%d %I:%M %p')
        sunset_dt = datetime.strptime(f"{today.strftime('%Y-%m-%d')} {sunset_time_str}", '%Y-%m-%d %I:%M %p')
    except ValueError:
        print("Warning: Failed to parse sunrise/sunset times. Using safe default.")
        current_hour = today.hour
        return current_hour < 7 or current_hour >= 19

    # Check if current time is BEFORE sunrise OR AFTER sunset
    is_night = today < sunrise_dt or today >= sunset_dt
    
    return is_night


# --- CORE LOGIC: API FETCH AND MAPPING ---

def fetch_real_time_weather():
    """Fetches real-time weather and astronomical data from WeatherAPI."""
    
    try:
        response = requests.get(WEATHER_URL, timeout=10)
        response.raise_for_status() 
        data = response.json()
        
        # 1. Extract necessary data points (WeatherAPI structure)
        current = data['current']
        astro = data['forecast']['forecastday'][0]['astro']
        
        # 2. Determine time-based features
        current_date = datetime.fromtimestamp(current['last_updated_epoch'])
        is_night = is_currently_night(astro['sunrise'], astro['sunset'])

        # 3. MAPPING: Map API fields to ML model features
        ml_features = {
            'temp': current['temp_c'], 
            'humidity': current['humidity'],
            'precip': current['precip_mm'], 
            'cloudcover': current['cloud'], 
            'visibility': min(current['vis_km'], 10.0), 
            'uvindex': current['uv'],
            'day_of_year': calculate_day_of_year(current_date),
            'is_night_time': is_night 
        }
        
        return {
            'status': 'success',
            'data': ml_features,
            'astronomy': {
                'sunrise': astro['sunrise'], 
                'sunset': astro['sunset']
            }
        }

    except requests.exceptions.RequestException as e:
        print(f"Weather API Error: {e}")
        return {'status': 'error', 'message': f'Could not connect to WeatherAPI: {e}'}
    except KeyError as e:
        print(f"Weather API Parsing Error: Missing key {e}")
        return {'status': 'error', 'message': f'Failed to parse weather data structure: Missing key {e}'}


def determine_lamp_action(all_inputs, model_clf: RandomForestClassifier, health_status: dict):
    """
    Implements the hierarchical control logic (Time -> Failsafe -> Safety -> Efficiency).
    """
    
    # --- 1. TIME CHECK (Step 1) ---
    if not all_inputs.get('is_night_time', False):
        return "OFF (Daytime)"

    # --- 2. FAILSAFE CHECK (HIGHEST PRIORITY) ---
    if health_status['is_overheated']:
        print(f"*** SYSTEM ALERT: OVERHEAT DETECTED ({health_status['current_temp_c']}°C). FORCING MAX OUTPUT. ***")
        return "MAX OUTPUT (System Failsafe: Overheat)"


    # --- 3. ML SAFETY CHECK (Weather Prediction) ---
    
    # Prepares features for the model (Only uses the 7 features the ML model was trained on)
    X_weather = pd.DataFrame([all_inputs])[FEATURES]
    
    # DEBUG CRITICAL: Print the order of features being passed to the model.
    print(f"Features passed to model (order MUST match training): {X_weather.columns.tolist()}") 
    
    # Predicts if MAX OUTPUT is required (1) or not (0) due to poor weather/visibility.
    max_light_required = model_clf.predict(X_weather)[0] 
    
    if max_light_required == 1:
        # Rainy or Foggy conditions predicted by ML model. Safety override.
        # DEBUG: Log the reason why ML model chose MAX OUTPUT
        print(f"*** ML SAFETY OVERRIDE: Weather features triggering MAX OUTPUT: {X_weather.to_dict('records')[0]} ***")
        return "MAX OUTPUT (Safety Override: Rain/Fog)"
        
    # --- 4. MOTION CHECK (EFFICIENCY - ONLY RUNS IF ML PREDICTS CLEAR WEATHER) ---
    
    if all_inputs['is_motion_detected']:
        # Motion detected -> Turn on Dimmer Light (Efficiency mode)
        return "DIMMER OUTPUT (Motion Detected)"
    else:
        # No motion detected, clear weather predicted -> Turn off light for max savings
        return "OFF (No Motion Detected)"


# --- FLASK ROUTE ---

@app.route('/api/control_lamp', methods=['POST'])
def control_lamp():
    if not model:
        return jsonify({'status': 'error', 'message': 'AI Model not loaded.'}), 500

    try:
        # 1. Get Motion Status
        client_data = request.get_json()
        is_motion_detected = client_data.get('is_motion_detected', False)

        # 2. Fetch Real-Time Weather Data
        weather_result = fetch_real_time_weather()
        if weather_result['status'] == 'error':
            return jsonify(weather_result), 500
        
        weather_features = weather_result['data']
        astronomy = weather_result['astronomy']
        
        # 3. Read System Health Sensor
        health_status = read_health_sensor()

        # 4. Combine ALL inputs into one dictionary for easy passing
        all_inputs = {
            **weather_features,
            'is_motion_detected': is_motion_detected
        }

        # 5. Determine Final Lamp Action using Hierarchical Logic
        lamp_action = determine_lamp_action(
            all_inputs, 
            model, 
            health_status
        )
        
        # 6. Return the result
        return jsonify({
            'status': 'success',
            'lamp_action': lamp_action,
            'system_health': health_status, 
            # We pass all combined inputs for the chart display:
            'weather_data_for_chart': all_inputs, 
            'inputs_used': {
                'is_night_time': all_inputs.get('is_night_time', False),
                'is_motion_detected': is_motion_detected,
                'location': LOCATION,
                'sunrise': astronomy['sunrise'],
                'sunset': astronomy['sunset']
            }
        })
        

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'An internal server error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
