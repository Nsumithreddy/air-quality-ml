import streamlit as st
import requests
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import time

# Initialize geolocator
geolocator = Nominatim(user_agent="air_quality_app")

# OpenWeatherMap API key
API_KEY = "f29f715e970ee8a71259a51c0b4b8b93"

# AQI classification as per base paper Table II
AQI_LEVELS = [
    (0, 50, "Excellent", "green"),
    (51, 100, "Good", "yellow"),
    (101, 150, "Lightly Polluted", "orange"),
    (151, 200, "Moderately Polluted", "red"),
    (201, 300, "Heavily Polluted", "purple"),
    (301, float("inf"), "Severely Polluted", "maroon")
]

def get_coordinates(location):
    """Validate location and get coordinates globally."""
    try:
        loc = geolocator.geocode(location)
        if loc:
            return loc.latitude, loc.longitude
        else:
            return None, None
    except:
        return None, None

def fetch_air_quality_data(lat, lon, retries=3):
    """Fetch air quality data from OpenWeatherMap API with retries."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                components = data["list"][0]["components"]
                return {
                    "pm2_5": components.get("pm2_5", 0),
                    "pm10": components.get("pm10", 0),
                    "o3": components.get("o3", 0),
                    "co": components.get("co", 0),
                    "no2": components.get("no2", 0),
                    "so2": components.get("so2", 0),
                    "timestamp": datetime.fromtimestamp(data["list"][0]["dt"])
                }
            else:
                st.warning(f"Attempt {attempt+1}: API request failed with status {response.status_code}: {response.text}")
                if attempt == retries - 1:
                    return {"error": f"API request failed after {retries} attempts: Status {response.status_code}, {response.text}"}
        except requests.RequestException as e:
            st.warning(f"Attempt {attempt+1}: API request error: {str(e)}")
            if attempt == retries - 1:
                return {"error": f"API request error after {retries} attempts: {str(e)}"}
        time.sleep(2)
    return {"error": "Failed to fetch air quality data after retries"}

def fetch_weather_data(lat, lon, retries=3):
    """Fetch weather data from OpenWeatherMap API with retries."""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature": data["main"].get("temp", 0),
                    "wind_speed": data["wind"].get("speed", 0)
                }
            else:
                st.warning(f"Attempt {attempt+1}: API request failed with status {response.status_code}: {response.text}")
                if attempt == retries - 1:
                    return {"error": f"API request failed after {retries} attempts: Status {response.status_code}, {response.text}"}
        except requests.RequestException as e:
            st.warning(f"Attempt {attempt+1}: API request error: {str(e)}")
            if attempt == retries - 1:
                return {"error": f"API request error after {retries} attempts: {str(e)}"}
        time.sleep(2)
    return {"error": "Failed to fetch weather data after retries"}

def calculate_aqi(pm25, pm10, o3):
    """Calculate AQI based on PM2.5, PM10, and O3 (simplified)."""
    pm25_breakpoints = [(0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)]
    for low, high, aqi_low, aqi_high in pm25_breakpoints:
        if low <= pm25 <= high:
            aqi_pm25 = ((aqi_high - aqi_low) / (high - low)) * (pm25 - low) + aqi_low
            break
    else:
        aqi_pm25 = 500
    return aqi_pm25

def fetch_historical_data(lat, lon, hours=48):
    """Fetch historical air quality and weather data with synthetic noise."""
    data = []
    current_time = datetime.now()
    np.random.seed(42)
    for i in range(hours):
        past_time = current_time - timedelta(hours=i)
        air_data = fetch_air_quality_data(lat, lon)
        if "error" in air_data:
            st.warning(air_data["error"])
            return pd.DataFrame()
        weather_data = fetch_weather_data(lat, lon)
        if "error" in weather_data:
            st.warning(weather_data["error"])
            return pd.DataFrame()
        if air_data:
            air_data["pm2_5"] += np.random.normal(0, 0.5)
            air_data["pm10"] += np.random.normal(0, 0.5)
            air_data["o3"] += np.random.normal(0, 0.5)
            weather_data["temperature"] += np.random.normal(0, 0.2)
            weather_data["wind_speed"] += np.random.normal(0, 0.1)
            air_data["pm2_5"] = max(0, air_data["pm2_5"])
            air_data["pm10"] = max(0, air_data["pm10"])
            air_data["o3"] = max(0, air_data["o3"])
            weather_data["temperature"] = max(-50, min(50, weather_data["temperature"]))
            weather_data["wind_speed"] = max(0, weather_data["wind_speed"])
            air_data["timestamp"] = past_time
            air_data["temperature"] = weather_data["temperature"]
            air_data["wind_speed"] = weather_data["wind_speed"]
            data.append(air_data)
        time.sleep(1)
    return pd.DataFrame(data)

def train_model(data):
    """Train XGBoost model with hyperparameter tuning and test RMSE."""
    if data.empty:
        return None, None, None
    data["hour"] = data["timestamp"].apply(lambda x: x.hour)
    X = data[["pm2_5", "pm10", "o3", "hour", "temperature", "wind_speed"]]
    y = data[["pm2_5", "pm10", "o3"]].shift(-4).dropna()
    X = X.iloc[:-4]
    if len(X) < 10:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1]
    }
    base_model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return model, train_rmse, test_rmse

def predict_next_4_hours(model, current_data, weather_data):
    """Predict air quality for the next 4 hours."""
    if model is None:
        return None
    current_hour = current_data["timestamp"].hour
    X = np.array([[current_data["pm2_5"], current_data["pm10"], current_data["o3"],
                   current_hour, weather_data["temperature"], weather_data["wind_speed"]]])
    prediction = model.predict(X)
    return {"pm2_5": prediction[0][0], "pm10": prediction[0][1], "o3": prediction[0][2]}

def plot_predicted_air_quality(prediction, predicted_aqi):
    """Plot predicted air quality data (PM2.5, PM10, O3) and AQI gauge using Plotly."""
    # Create a subplot with two sections: Bar chart for pollutants, Gauge for AQI
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Predicted Air Quality (Next 4 Hours)", "Predicted AQI Level"),
                        specs=[[{"type": "xy"}], [{"type": "xy"}]])

    # Bar chart for predicted PM2.5, PM10, O3
    pollutants = ["PM2.5", "PM10", "O3"]
    values = [prediction["pm2_5"], prediction["pm10"], prediction["o3"]]
    fig.add_trace(go.Bar(
        x=values,
        y=pollutants,
        orientation="h",
        marker=dict(color=["blue", "green", "red"]),
        text=[f"{val:.2f} µg/m³" for val in values],
        textposition="auto"
    ), row=1, col=1)

    # AQI gauge for predicted AQI
    for low, high, label, color in AQI_LEVELS:
        if low <= predicted_aqi <= high:
            fig.add_trace(go.Bar(
                y=[label],
                x=[predicted_aqi],
                orientation="h",
                marker=dict(color=color),
                text=[f"AQI: {int(predicted_aqi)} ({label})"],
                textposition="auto"
            ), row=2, col=1)
            break

    # Update layout for better visualization
    fig.update_layout(height=600, showlegend=False, xaxis2=dict(range=[0, 500]))
    fig.update_xaxes(title_text="Concentration (µg/m³)", row=1, col=1)
    fig.update_xaxes(title_text="AQI Value", row=2, col=1)
    return fig

# Streamlit app
st.title("Real-Time Air Quality Monitoring")
st.write("Enter any city or location to monitor air quality and get 4-hour predictions.")

location = st.text_input("Enter location (e.g., Hyderabad, London):", "Chennai")
if st.button("Fetch Data"):
    lat, lon = get_coordinates(location)
    if lat is None or lon is None:
        st.error(f"Invalid location: {location}. Please enter a valid city or location.")
    else:
        st.success(f"Fetching data for {location}...")
        air_data = fetch_air_quality_data(lat, lon)
        if "error" in air_data:
            st.error(air_data["error"])
        else:
            weather_data = fetch_weather_data(lat, lon)
            if "error" in weather_data:
                st.error(weather_data["error"])
            else:
                # Display current air quality data as text
                st.write("### Current Air Quality")
                st.write(f"**PM2.5**: {air_data['pm2_5']:.2f} µg/m³")
                st.write(f"**PM10**: {air_data['pm10']:.2f} µg/m³")
                st.write(f"**O3**: {air_data['o3']:.2f} µg/m³")
                st.write(f"**CO**: {air_data['co']:.2f} µg/m³")
                st.write(f"**NO2**: {air_data['no2']:.2f} µg/m³")
                st.write(f"**SO2**: {air_data['so2']:.2f} µg/m³")
                st.write(f"**Temperature**: {weather_data['temperature']:.2f} °C")
                st.write(f"**Wind Speed**: {weather_data['wind_speed']:.2f} m/s")

                # Fetch historical data for training
                historical_data = fetch_historical_data(lat, lon)
                if not historical_data.empty:
                    # Train model and get predictions
                    model, train_rmse, test_rmse = train_model(historical_data)
                    if model:
                        st.write(f"**Train RMSE**: {train_rmse:.2f} (lower is better)")
                        st.write(f"**Test RMSE**: {test_rmse:.2f} (lower is better)")
                        prediction = predict_next_4_hours(model, air_data, weather_data)
                        if prediction:
                            predicted_aqi = calculate_aqi(prediction["pm2_5"], prediction["pm10"], prediction["o3"])
                            # Display predicted data as text
                            st.write("### 4-Hour Air Quality Prediction")
                            st.write(f"**PM2.5**: {prediction['pm2_5']:.2f} µg/m³")
                            st.write(f"**PM10**: {prediction['pm10']:.2f} µg/m³")
                            st.write(f"**O3**: {prediction['o3']:.2f} µg/m³")
                            for low, high, label, color in AQI_LEVELS:
                                if low <= predicted_aqi <= high:
                                    st.write(f"**Predicted AQI**: {int(predicted_aqi)} ({label})")
                                    break
                            # Interactive visualization for predicted data
                            fig = plot_predicted_air_quality(prediction, predicted_aqi)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to train the model.")
                else:
                    st.warning("Could not fetch historical data.")