from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# -------------------- Page & constants --------------------
st.set_page_config(page_title="Air Quality Monitor", layout="wide")

API_KEY = st.secrets.get("OPENWEATHER_API_KEY")
if not API_KEY:
    st.error("Missing OPENWEATHER_API_KEY in secrets. Add it before running.")

# AQI ranges and simple labels for clarity
AQI_BANDS = [
    (0, 50, "Good", "✅ Good", "green"),
    (51, 100, "Moderate", "🟡 Moderate", "yellow"),
    (101, 150, "Unhealthy for Sensitive", "🟠 Unhealthy (Sensitive)", "orange"),
    (151, 200, "Unhealthy", "🔴 Unhealthy", "red"),
    (201, 300, "Very Unhealthy", "🟣 Very Unhealthy", "purple"),
    (301, 500, "Hazardous", "🟤 Hazardous", "maroon"),
]

# US EPA PM2.5 breakpoints for index calc
PM25_BREAKS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)
]
# Rough human-readable bands
PM10_CATS = [(0, 50, "Good"), (51, 100, "Moderate"), (101, 250, "Unhealthy"), (251, 600, "Hazardous")]
O3_CATS   = [(0, 60, "Good"), (61, 120, "Moderate"), (121, 180, "Unhealthy"), (181, 400, "Hazardous")]

# Geocoder
geolocator = Nominatim(user_agent="air_quality_monitor", timeout=10)

@st.cache_data(ttl=86400)
def geocode(place: str):
    try:
        loc = geolocator.geocode(place)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        pass
    return None, None

# -------------------- OpenWeather helpers (HTTPS, cached) --------------------
@st.cache_data(ttl=300)
def fetch_current_air(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return {"error": f"Air API {r.status_code}: {r.text[:160]}"}
        item = r.json()["list"][0]
        comp = item["components"]
        return {
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "pm2_5": float(comp.get("pm2_5", 0.0)),
            "pm10": float(comp.get("pm10", 0.0)),
            "o3": float(comp.get("o3", 0.0)),
            "co": float(comp.get("co", 0.0)),
            "no2": float(comp.get("no2", 0.0)),
            "so2": float(comp.get("so2", 0.0)),
        }
    except requests.RequestException as e:
        return {"error": f"Air API error: {e}"}

@st.cache_data(ttl=300)
def fetch_current_weather(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return {"error": f"Weather API {r.status_code}: {r.text[:160]}"}
        d = r.json()
        return {
            "temperature": float(d.get("main", {}).get("temp", 0.0)),
            "wind_speed": float(d.get("wind", {}).get("speed", 0.0)),
        }
    except requests.RequestException as e:
        return {"error": f"Weather API error: {e}"}

@st.cache_data(ttl=900)
def fetch_history(lat: float, lon: float, hours: int = 48) -> pd.DataFrame:
    """Real OpenWeather history (no synthetic noise). If unavailable, return empty."""
    end = int(datetime.now(tz=timezone.utc).timestamp())
    start = end - hours * 3600
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
    )
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        js = r.json()
        if "list" not in js or not js["list"]:
            return pd.DataFrame()
        rows = []
        for item in js["list"]:
            comp = item.get("components", {})
            rows.append({
                "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                "pm2_5": float(comp.get("pm2_5", 0.0)),
                "pm10": float(comp.get("pm10", 0.0)),
                "o3": float(comp.get("o3", 0.0)),
            })
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    except requests.RequestException:
        return pd.DataFrame()

# -------------------- Readable categories --------------------
def aqi_from_pm25(pm25: float) -> float:
    for low, high, aqi_low, aqi_high in PM25_BREAKS:
        if low <= pm25 <= high:
            return ((aqi_high - aqi_low) / (high - low)) * (pm25 - low) + aqi_low
    return 500.0

def band_from_aqi(aqi: float):
    for low, high, short, label, color in AQI_BANDS:
        if low <= aqi <= high:
            return short, label, color
    return "Hazardous", "🟤 Hazardous", "maroon"

# -------------------- Modeling (no noise) --------------------
def train_model(hist: pd.DataFrame, weather_snap: dict):
    if hist is None or hist.empty or len(hist) < 24:
        return None, None, None
    df = hist.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["temperature"] = float(weather_snap.get("temperature", 0.0))
    df["wind_speed"] = float(weather_snap.get("wind_speed", 0.0))

    X = df[["pm2_5", "pm10", "o3", "hour", "temperature", "wind_speed"]]
    y = df[["pm2_5", "pm10", "o3"]].shift(-4).dropna()
    X = X.iloc[: len(y)]
    if len(X) < 20:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    y_tr, y_te = model.predict(X_train), model.predict(X_test)
    tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_tr)))
    te_rmse = float(np.sqrt(mean_squared_error(y_test, y_te)))
    return model, tr_rmse, te_rmse

def predict_next_4h(model, curr_air: dict, curr_weather: dict):
    if model is None:
        return None
    cur_hour = curr_air["timestamp"].hour
    X = np.array([[curr_air["pm2_5"], curr_air["pm10"], curr_air["o3"], cur_hour,
                   float(curr_weather.get("temperature", 0.0)), float(curr_weather.get("wind_speed", 0.0))]])
    pred = model.predict(X)[0]
    return {"pm2_5": float(pred[0]), "pm10": float(pred[1]), "o3": float(pred[2])}

# -------------------- Plotting --------------------
def plot_pred(pred: dict, aqi_val: float):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],         # give AQI bar its own space
        vertical_spacing=0.25,            # more gap between charts
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    # --- Top: pollutants ---
    pollutants = ["PM2.5", "PM10", "O3"]
    values = [pred["pm2_5"], pred["pm10"], pred["o3"]]
    fig.add_trace(
        go.Bar(
            x=values, y=pollutants, orientation="h",
            text=[f"{v:.1f} µg/m³" for v in values],
            textposition="outside",        # move labels outside bars
            cliponaxis=False,
        ),
        row=1, col=1,
    )

    # --- Bottom: AQI band ---
    _, aqi_label, _ = band_from_aqi(aqi_val)
    fig.add_trace(
        go.Bar(
            y=[aqi_label], x=[aqi_val], orientation="h",
            text=[f"AQI {int(aqi_val)}"], textposition="outside",
            cliponaxis=False,
        ),
        row=2, col=1,
    )

    # Layout tweaks to prevent collisions
    fig.update_layout(
        height=720,
        showlegend=False,
        bargap=0.3,
        margin=dict(t=40, r=30, b=80, l=60),
    )
    fig.update_xaxes(title_text="Concentration (µg/m³)", row=1, col=1, title_standoff=12)
    fig.update_xaxes(title_text="AQI Value", row=2, col=1, title_standoff=12, range=[0, 500])
    fig.update_yaxes(automargin=True)  # ensure category labels don't clash

    return fig


# -------------------- UI --------------------
st.title("Real-Time Air Quality Monitoring")
st.caption("Type a location → fetch live OpenWeather data → get AQI band and optional +4h prediction.")

col_in = st.columns([2, 1])
with col_in[0]:
    place = st.text_input("Enter location (e.g., Hyderabad, London)", "Chennai")
with col_in[1]:
    coord_mode = st.toggle("Use latitude/longitude instead")

if coord_mode:
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=13.0827, format="%.6f")
    with c2:
        lon = st.number_input("Longitude", value=80.2707, format="%.6f")
else:
    lat, lon = (None, None)

if st.button("Fetch Data", use_container_width=True):
    if coord_mode and (lat is None or lon is None):
        st.error("Enter valid coordinates.")
        st.stop()
    if not coord_mode:
        lat, lon = geocode(place)
    if lat is None or lon is None:
        st.error("Could not find that place. Try a more specific name.")
        st.stop()

    air = fetch_current_air(lat, lon)
    if "error" in air: st.error(air["error"]); st.stop()

    weather = fetch_current_weather(lat, lon)
    if "error" in weather: st.error(weather["error"]); st.stop()

    # Current section
    st.subheader("Current Air Quality")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("PM2.5 (µg/m³)", f"{air['pm2_5']:.1f}")
        st.metric("CO (µg/m³)", f"{air['co']:.1f}")
    with c2:
        st.metric("PM10 (µg/m³)", f"{air['pm10']:.1f}")
        st.metric("NO2 (µg/m³)", f"{air['no2']:.1f}")
    with c3:
        st.metric("O3 (µg/m³)", f"{air['o3']:.1f}")
        st.metric("SO2 (µg/m³)", f"{air['so2']:.1f}")

    aqi_now = aqi_from_pm25(air['pm2_5'])
    _, band_label, _ = band_from_aqi(aqi_now)
    st.info(f"**Current AQI:** {int(aqi_now)} — {band_label}")
    st.caption(f"Weather — Temp: {weather['temperature']:.1f} °C | Wind: {weather['wind_speed']:.1f} m/s")

    # History for prediction
    hist = fetch_history(lat, lon, hours=48)
    if hist.empty:
        st.warning("Historical data not available here. Showing current only (no prediction).")
        snap = pd.DataFrame([{**air, **weather}])
        st.download_button("Download current snapshot (CSV)", snap.to_csv(index=False).encode(), file_name="current_air_quality.csv")
        st.stop()

    st.download_button("Download history (CSV)", hist.to_csv(index=False).encode(), file_name="history_air_quality.csv")

    model, tr_rmse, te_rmse = train_model(hist, weather)
    if model is None:
        st.warning("Not enough history to train a predictor here.")
        st.stop()

    pred = predict_next_4h(model, air, weather)
    if not pred:
        st.warning("Prediction unavailable.")
        st.stop()

    aqi_pred = aqi_from_pm25(pred['pm2_5'])
    _, pred_label, _ = band_from_aqi(aqi_pred)

    st.subheader("Prediction (+4 hours)")
    p1, p2, p3 = st.columns(3)
    p1.metric("PM2.5", f"{pred['pm2_5']:.1f} µg/m³")
    p2.metric("PM10", f"{pred['pm10']:.1f} µg/m³")
    p3.metric("O3", f"{pred['o3']:.1f} µg/m³")
    st.info(f"**Predicted AQI:** {int(aqi_pred)} — {pred_label}")

    st.caption(f"Model RMSE — Train: {tr_rmse:.2f} | Test: {te_rmse:.2f} (lower is better)")

    fig = plot_pred(pred, aqi_pred)
    st.plotly_chart(fig, use_container_width=True)