from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import atexit

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load or train models
def load_or_train_models():
    try:
        with open('models/status_model.pkl', 'rb') as f:
            status_model = pickle.load(f)
        with open('models/moisture_model.pkl', 'rb') as f:
            moisture_model = pickle.load(f)
        print("Models loaded successfully")
        return status_model, moisture_model
    except Exception as e:
        print(f"Error loading models: {e}. Attempting to train new models...")
        try:
            from train_model import train_models
            return train_models()
        except Exception as train_error:
            print(f"Training failed: {train_error}")
            exit()

status_model, moisture_model = load_or_train_models()

# Data storage with additional fields
last_data = {
    "temperature_now": 25.0,
    "humidity_now": 60.0,
    "rtc": "00:00:00",
    "elapsed_time": "00:00:00",
    "heater_status": "OFF",
    "fan_status": "OFF",
    "clove_status": "Basah",
    "moisture_content": 0.0,
    "target_moisture": 16,
    "target_time": 4,
    "process_finish": False,
    "last_update": datetime.now().isoformat()
}

# Load config if exists
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        last_data.update(config)
    print("Loaded config from file")
except FileNotFoundError:
    print("No config file found, using defaults")

def save_config():
    with open('config.json', 'w') as f:
        json.dump({
            "target_moisture": last_data['target_moisture'],
            "target_time": last_data['target_time']
        }, f)

# State persistence functions
def save_last_state():
    if 'last_data' in globals():
        with open('last_state.json', 'w') as f:
            json.dump({
                'temperature': last_data['temperature_now'],
                'humidity': last_data['humidity_now'],
                'timestamp': datetime.now().isoformat()
            }, f)

def load_last_state():
    try:
        with open('last_state.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Load data saat startup
last_state = load_last_state()
if last_state:
    last_data.update({
        'temperature_now': last_state['temperature'],
        'humidity_now': last_state['humidity']
    })

# Register shutdown handler
atexit.register(save_last_state)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # === START OF MODIFIED FALLBACK LOGIC ===
        # Validate required fields
        required_fields = ['temperature_now', 'humidity_now', 'time_remaining']
        if not all(field in data for field in required_fields):
            return jsonify({
                "status": "error",
                "message": "Data sensor tidak lengkap",
                "required_fields": required_fields
            }), 400
            
        # Convert values with explicit validation
        try:
            temp = float(data['temperature_now'])
            hum = float(data['humidity_now'])
            time_remain = float(data['time_remaining'])
        except (ValueError, TypeError) as e:
            return jsonify({
                "status": "error",
                "message": f"Data sensor tidak valid: {str(e)}"
            }), 400

        # Optional fields with safe defaults
        rtc_time = data.get('rtc', "00:00:00")
        elapsed_time = data.get('elapsed_time', "00:00:00")
        # === END OF MODIFIED FALLBACK LOGIC ===

        # AI Prediction (unchanged)
        X_pred = [[hum, temp, time_remain]]
        status_kering = status_model.predict(X_pred)[0]
        moisture_content = round(float(moisture_model.predict(X_pred)[0]), 2)

        # Process control logic (unchanged)
        process_finish = (moisture_content <= last_data['target_moisture']) or (time_remain <= 0)
        heater_status = "ON" if (not process_finish and temp < 50) else "OFF"
        fan_status = "ON" if not process_finish else "OFF"

        # Update last_data (unchanged)
        last_data.update({
            "temperature_now": temp,
            "humidity_now": hum,
            "rtc": rtc_time,
            "elapsed_time": elapsed_time,
            "heater_status": heater_status,
            "fan_status": fan_status,
            "clove_status": status_kering,
            "moisture_content": moisture_content,
            "process_finish": process_finish,
            "last_update": datetime.now().isoformat()
        })

        # Save to CSV log (unchanged)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "temperature": temp,
            "humidity": hum,
            "moisture_pred": moisture_content,
            "status": status_kering,
            "heater": heater_status,
            "fan": fan_status,
            "target_moisture": last_data['target_moisture'],
            "target_time": last_data['target_time']
        }
        pd.DataFrame([log_entry]).to_csv(
            'logs/data.csv',
            mode='a',
            header=not os.path.exists('logs/data.csv'),
            index=False
        )

        # Save current state
        save_last_state()

        return jsonify({
            "status": "success",
            "temperature_now": temp,
            "humidity_now": hum,
            "moisture_content": moisture_content,
            "heater_status": heater_status,
            "fan_status": fan_status,
            "clove_status": status_kering,
            "process_finish": process_finish,
            "rtc": rtc_time,
            "elapsed_time": elapsed_time
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get-data', methods=['GET'])
def get_data():
    return jsonify(last_data)

@app.route('/update-target', methods=['POST'])
def update_target():
    try:
        data = request.json
        last_data['target_moisture'] = int(data.get('target_moisture', last_data['target_moisture']))
        last_data['target_time'] = int(data.get('target_time', last_data['target_time']))
        save_config()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)