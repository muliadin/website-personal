from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

app = Flask(__name__)

def load_models():
    global status_model, moisture_model, time_model
    with open('status_model.pkl', 'rb') as f:
        status_model = pickle.load(f)
    with open('moisture_model.pkl', 'rb') as f:
        moisture_model = pickle.load(f)
    with open('time_model.pkl', 'rb') as f:
        time_model = pickle.load(f)

load_models()

last_data = {
    "temperature_now": 0,
    "humidity_now": 0,
    "rtc": "00:00:00",
    "elapsed_time": "00:00:00",
    "heater_status": "OFF",
    "fan_status": "OFF",
    "clove_status": "-",
    "target_moisture": 13,
    "target_time": 9800,
    "process_finish": False
}
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        df = pd.read_csv('data_log.csv')
        if len(df) < 10:
            return jsonify({"status": "Gagal", "message": "Data belum cukup untuk training"})

        X = df[['humidity_now', 'temperature_now', 'time_remaining']]

        # Train status model
        y_status = df['status_kering']
        new_status_model = DecisionTreeClassifier()
        new_status_model.fit(X, y_status)

        # Train moisture model
        y_moisture = df['moisture_content']
        new_moisture_model = DecisionTreeRegressor()
        new_moisture_model.fit(X, y_moisture)

        # Train time model
        y_time = df['target_time']
        new_time_model = DecisionTreeRegressor()
        new_time_model.fit(X, y_time)

        # Simpan model ke file
        with open('status_model.pkl', 'wb') as f:
            pickle.dump(new_status_model, f)
        with open('moisture_model.pkl', 'wb') as f:
            pickle.dump(new_moisture_model, f)
        with open('time_model.pkl', 'wb') as f:
            pickle.dump(new_time_model, f)

        # Reload model ke memory
        load_models()

        # TEST PREDIKSI — contoh 5 data acak dari data_train.csv
        test_samples = X.sample(5)
        pred_time = time_model.predict(test_samples)

        print("=== HASIL PREDIKSI TARGET_TIME SETELAH RETRAIN ===")
        for idx, t in enumerate(pred_time):
            print(f"Sample {idx+1}: Prediksi target_time = {t} detik")

        return jsonify({"status": "Berhasil", "message": "Model berhasil diretrain & di-reload. Lihat console log hasil prediksi."})

    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)})
