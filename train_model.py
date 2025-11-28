import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- 1. DATA PREPARATION AND MODEL TRAINING ---
print("Starting Model Training...")

try:
    df = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("Error: 'weather.csv' not found. Please ensure the file is in the working directory.")
    exit()

# Feature Engineering
df['datetime'] = pd.to_datetime(df['datetime'])
df['day_of_year'] = df['datetime'].dt.dayofyear
df['max_light_required'] = df['conditions'].str.lower().str.contains('rain|fog').astype(int)

FEATURES = ['humidity', 'cloudcover', 'visibility', 'uvindex', 'day_of_year', 'temp', 'precip']
X = df[FEATURES]
y = df['max_light_required']

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
model_clf.fit(X, y)

# --- 2. SAVE THE TRAINED MODEL ---
MODEL_FILE = 'street_lamp_model.joblib'
joblib.dump(model_clf, MODEL_FILE)

print("AI Model Trained Successfully.")
print(f"Model saved to: {MODEL_FILE}")
print("-" * 50)