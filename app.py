import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# =============================
# CONFIG
# =============================
DATA_FILE = "data/weather_dataset.csv"
MODEL_FILE = "models/transformer_model.keras"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

st.set_page_config(layout="wide")
st.title("🌦️ Real-Time Weather Transformer AI")

# =============================
# LOCATION SEARCH
# =============================
@st.cache_data(ttl=3600)
def search_location(query):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5"
    return requests.get(url).json().get("results", [])

# =============================
# HISTORICAL WEATHER API
# =============================
@st.cache_data(ttl=86400)
def fetch_historical_weather(lat, lon, start_date, end_date):

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        "&timezone=auto"
    )

    data = requests.get(url).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relativehumidity_2m"],
        "wind": data["hourly"]["windspeed_10m"],
        "pressure": data["hourly"]["surface_pressure"],
    })

    return df

# =============================
# FORECAST WEATHER
# =============================
def fetch_latest_weather(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        "&forecast_days=3&timezone=auto"
    )

    data = requests.get(url).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relativehumidity_2m"],
        "wind": data["hourly"]["windspeed_10m"],
        "pressure": data["hourly"]["surface_pressure"],
    })

    return df

# =============================
# UPDATE DATASET SMARTLY
# =============================
def update_dataset(lat, lon):

    today = datetime.utcnow().date()

    # If dataset exists -> load
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["time"])
        last_date = df["time"].max().date()
    else:
        df = pd.DataFrame()
        last_date = today - timedelta(days=365)

    # Fetch missing historical
    if last_date < today:

        start = last_date + timedelta(days=1)
        st.info(f"Loading historical data {start} → {today}")

        hist_df = fetch_historical_weather(
            lat, lon,
            start.isoformat(),
            today.isoformat()
        )

        df = pd.concat([df, hist_df])

    # Fetch latest short-term forecast (acts as recent observed)
    latest = fetch_latest_weather(lat, lon)

    # Remove future timestamps
    latest = latest[latest["time"] <= pd.Timestamp.now()]

    df = pd.concat([df, latest])
    df = df.drop_duplicates(subset="time")
    df = df.sort_values("time")

    df.to_csv(DATA_FILE, index=False)

    return df

# =============================
# TRANSFORMER MODEL
# =============================
def build_transformer(seq_len, features):

    inputs = layers.Input(shape=(seq_len, features))

    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = layers.LayerNormalization()(inputs + attn)

    ff = layers.Dense(128, activation="relu")(x)
    ff = layers.Dense(features)(ff)

    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model

# =============================
# SEQUENCE PREP
# =============================
def prepare_sequences(data, seq_len=24):

    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])

    return np.array(X), np.array(y)

# =============================
# LOCATION INPUT
# =============================
query = st.text_input("📍 Enter City")

lat, lon = None, None

if query:
    results = search_location(query)

    if results:
        options = {
            f"{r['name']}, {r.get('country','')}":
            (r["latitude"], r["longitude"])
            for r in results
        }

        selected = st.selectbox("Select Location", options.keys())
        lat, lon = options[selected]

# =============================
# MAIN PIPELINE
# =============================
if lat and lon:

    df = update_dataset(lat, lon)

    df = df.set_index("time")
    df = df.dropna()

    st.success(f"Dataset updated until {df.index.max()}")
    st.write(f"Total samples: {len(df)}")

    # =============================
    # SCALE
    # =============================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    seq_len = 24
    X, y = prepare_sequences(scaled, seq_len)

    if len(X) < 100:
        st.warning("Need more history data")
        st.stop()

    # =============================
    # TRAIN / LOAD MODEL
    # =============================
    if os.path.exists(MODEL_FILE):
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        model = build_transformer(seq_len, df.shape[1])
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        model.save(MODEL_FILE)

    # =============================
    # FORECAST
    # =============================
    last_seq = X[-1].reshape(1, seq_len, df.shape[1])

    forecast_scaled = []
    seq = last_seq.copy()

    for _ in range(24):
        pred = model.predict(seq, verbose=0)[0][0]

        new_step = seq[0, -1].copy()
        new_step[0] = pred

        forecast_scaled.append(new_step)

        seq = np.append(seq[:, 1:, :], [[new_step]], axis=1)

    forecast_scaled = np.array(forecast_scaled)
    forecast_real = scaler.inverse_transform(forecast_scaled)
    forecast_temp = forecast_real[:, 0]

    future_time = pd.date_range(datetime.now(), periods=24, freq="H")

    forecast_temp = [round(float(x), 2) for x in forecast_temp]

    # =============================
    # GRAPH
    # =============================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index[-72:],
        y=df["temp"].tail(72),
        name="Past Temperature"
    ))

    fig.add_trace(go.Scatter(
        x=future_time,
        y=forecast_temp,
        name="Forecast"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # TABLE
    # =============================
    st.subheader("Next 24 Hour Forecast")

    forecast_df = pd.DataFrame({
        "Time": future_time,
        "Temperature °C": forecast_temp
    })

    st.dataframe(forecast_df)