import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import numpy as np 

# --- CONFIGURATION ---
DATA_FILE = 'weather.csv'
MODEL_FILE = 'street_lamp_model.joblib'

# The features used by the model for prediction (must match app.py's FEATURES list)
# We exclude is_night_time and is_motion_detected because app.py handles those explicitly.
FEATURES = ['humidity', 'cloudcover', 'visibility', 'uvindex', 'day_of_year', 'temp', 'precip']

# --- 1. HELPER FUNCTIONS ---

def calculate_day_of_year(dt):
    """Calculates the day of the year (1-366)."""
    # This is kept as it's needed to generate the 'day_of_year' feature from the datetime column.
    return dt.timetuple().tm_yday

# --- 2. LOAD AND FEATURE ENGINEERING ---
print("1. Loading and feature engineering data...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Ensure it is in the same directory.")
    exit()

# Prepare datetime for feature extraction
df['datetime_obj'] = pd.to_datetime(df['datetime'], errors='coerce')
df['day_of_year'] = df['datetime_obj'].apply(calculate_day_of_year)


# 2.1. Define the binary target variable (Safety Override Prediction)
# The model's sole job is to predict if MAX LIGHT is required due to poor visibility/weather.
# 1 = MAX OUTPUT (Rainy or Foggy), 0 = DIMMER/OFF (Otherwise, i.e., good weather)
df['max_light_required'] = df['conditions'].str.lower().str.contains('rain|fog').astype(int)

# --- 3. TRAIN AND SAVE ---

X = df[FEATURES]
y = df['max_light_required']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X, y)

print("\n2. Model training complete (Binary Safety Prediction on 7 features).")

# Save the model
joblib.dump(model, MODEL_FILE)
print(f"3. Successfully saved model to {MODEL_FILE}")

print("\n\n✅ TRAINING COMPLETE! The model is now consistent with app.py's hierarchical control logic.")
