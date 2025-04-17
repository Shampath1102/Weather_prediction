from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained models
temperature_model = joblib.load("temp_model.pkl")
wind_model = joblib.load("wind_model.pkl")
climate_model = joblib.load("climate_model.pkl")
climate_encoder = joblib.load("climate_encoder.pkl")

# Extract features from datetime input (MATCHING training features)
def extract_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    features = {
        "dayofyear": dt.timetuple().tm_yday,
        "month": dt.month,
        "weekday": dt.weekday(),
        # Using average values from historical data
        "temp_min": 25.0,
        "temp_max": 32.0,
        "pressure": 1013.0,
        "precipitation": 0.0
    }
    
    return pd.DataFrame([[
        features["dayofyear"], 
        features["month"], 
        features["weekday"],
        features["temp_min"],
        features["temp_max"],
        features["pressure"],
        features["precipitation"]
    ]], columns=["dayofyear", "month", "weekday", "temp_min", "temp_max", "pressure", "precipitation"])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_date = request.form["datetime"]
        features = extract_features(input_date)

        # Predict values
        temp = temperature_model.predict(features)[0]
        climate_encoded = climate_model.predict(features)[0]
        climate = climate_encoder.inverse_transform([climate_encoded])[0]

        prediction = {
            "Temperature (°C)": round(temp, 2),
            "Climate": climate
        }

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
