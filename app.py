import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os
from datetime import datetime

# =============================
# CONFIG
# =============================
DATA_FILE = "data/weather_dataset.csv"
MODEL_FILE = "models/transformer_model.keras"

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

st.set_page_config(layout="wide")
st.title("🌦️ Real-Time Transformer Weather AI")

# =============================
# LOCATION SEARCH
# =============================

@st.cache_data(ttl=3600)
def search_location(query):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5"
    return requests.get(url).json().get("results", [])

# =============================
# REAL TIME WEATHER FETCH
# =============================

def fetch_latest_weather(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        f"&forecast_days=3&timezone=auto"
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
# DATA STORAGE
# =============================

def update_local_dataset(new_df):

    if os.path.exists(DATA_FILE):
        old_df = pd.read_csv(DATA_FILE, parse_dates=["time"])
        df = pd.concat([old_df, new_df])
    else:
        df = new_df

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
# PREPARE DATA
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

    st.info("Fetching latest weather data...")

    new_data = fetch_latest_weather(lat, lon)
    df = update_local_dataset(new_data)

    df = df.set_index("time")
    df = df.dropna()

    st.subheader("📊 Latest Training Dataset")
    st.dataframe(df.tail(48))

    # =============================
    # SCALING
    # =============================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    seq_len = 24
    X, y = prepare_sequences(scaled, seq_len)

    if len(X) < 50:
        st.warning("Not enough data yet")
        st.stop()

    # =============================
    # TRAIN OR LOAD MODEL
    # =============================

    if os.path.exists(MODEL_FILE):

        model = tf.keras.models.load_model(MODEL_FILE)
        st.success("Loaded Existing AI Model")

    else:
        st.info("Training New AI Model...")

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

    # =============================
    # TIME
    # =============================

    future_time = pd.date_range(
        start=datetime.now(),
        periods=24,
        freq="H"
    )

    forecast_clean = [round(float(x), 2) for x in forecast_temp]

    # =============================
    # GRAPH
    # =============================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index[-48:],
        y=df["temp"].tail(48),
        name="Past Temp"
    ))

    fig.add_trace(go.Scatter(
        x=future_time,
        y=forecast_clean,
        name="Forecast Temp"
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # TABLE
    # =============================

    forecast_df = pd.DataFrame({
        "Time": future_time,
        "Temperature °C": forecast_clean
    })

    st.subheader("🌡️ Next 24 Hour Forecast")
    st.dataframe(forecast_df)