HELIOX - A Smart Street Lamp AI Control Dashboard
This repository contains the complete source code for a Smart Street Lamp Control Dashboard
- featuring an AI backend (Flask) and a single-page HTML frontend (with CSS and JavaScript bundled for portability)

The system demonstrates intelligent traffic and emergency decision-making by loading a simulated machine learning model (street_lamp_model.joblib) to control the lamp's brightness.

⚠️ CRITICAL SETUP: Project Structure to ensure the Flask server runs correctly and can serve both your HTML pages (as templates) and your assets (as static files), the project directory must be reorganized.
Action Required for Examiner: Please perform the following steps before running app.py:
1. Create a folder named templates.
2. Move index.html and overview.html into the templates folder.
Your final project structure should look like this:
project_root/
- app.py
- getLampAction.js
- street_lamp_model.joblib
- train_model.py
- weather.csv
- nighttime_withoutcar.mp4         (Clear Road)
- accidents.mp4    (Crash/Emergency)
- Night_withcardetected.mp4     (Vehicle Detected)
- templates
     - index.html
     - overview.html
   
🚀 How to Set Up and Run the Application
1. Prerequisites: You must have Python 3 installed and the following libraries available. We recommend using your Anaconda Prompt or terminal:
- Bash:
- pip install flask pandas scikit-learn joblib
2. Training the Model (If necessary)The AI decision logic relies on the model file (street_lamp_model.joblib).
- Run the training script (this only needs to be done once):
- Bash:
- python train_model.py
3. Start the Backend ServerRun the Flask server from the project's root directory:
- Bash:
- python app.py
- You should see output confirming the server is ready:* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
- Keep this terminal window open.
4. Access the Dashboard
- Open your web browser and navigate directly to the local server URL:http://127.0.0.1:5000/
   
⚙️Testing and Automated Scenarios (3-Step Loop)The dashboard is configured with an automated demo sequence that begins immediately upon loading. 
- It cycles through the following 3 scenarios every few seconds, demonstrating the lamp's response to traffic and emergency conditions.
  
