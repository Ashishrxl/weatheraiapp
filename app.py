import streamlit as st
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Transformer Weather AI", layout="wide")

st.title("🌦️ Multi-Variable Transformer Weather Forecast AI")
st.markdown("### 🤖 AI Powered Weather Prediction")

# =============================
# CACHE FUNCTIONS
# =============================

@st.cache_data(ttl=3600)
def search_location(query):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5"
    res = requests.get(url).json()
    return res.get("results", [])


@st.cache_data(ttl=3600)
def fetch_weather_data(lat, lon):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date=2024-01-01&end_date=2024-12-31"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        f"&timezone=auto"
    )

    return requests.get(url).json()

# =============================
# TRANSFORMER MODEL
# =============================

def build_transformer_model(seq_len, num_features):
    inputs = layers.Input(shape=(seq_len, num_features))

    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=32)(inputs, inputs)

    x = layers.LayerNormalization()(inputs + attention)

    ff = layers.Dense(128, activation="relu")(x)
    ff = layers.Dense(num_features)(ff)

    x = layers.LayerNormalization()(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model

# =============================
# DATA PREPARATION
# =============================

def prepare_data(data, seq_len=24):
    features = data.values

    X, y = [], []

    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(features[i+seq_len][0])  # Predict temperature

    return np.array(X), np.array(y)

# =============================
# LOCATION INPUT
# =============================

query = st.text_input("📍 Type City Name")

selected_lat = None
selected_lon = None

if query:
    suggestions = search_location(query)

    if suggestions:
        options = {
            f"{s['name']}, {s.get('country','')} (Lat:{s['latitude']}, Lon:{s['longitude']})":
            (s["latitude"], s["longitude"])
            for s in suggestions
        }

        choice = st.selectbox("Select Location", list(options.keys()))

        selected_lat, selected_lon = options[choice]

        st.success(f"Latitude: {selected_lat} | Longitude: {selected_lon}")

# =============================
# MAIN FORECAST
# =============================

if selected_lat and selected_lon:

    weather = fetch_weather_data(selected_lat, selected_lon)

    hourly = weather["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temp": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "wind": hourly["windspeed_10m"],
        "pressure": hourly["surface_pressure"],
    })

    df = df.dropna()
    df = df.set_index("time")

    st.subheader("📊 Training Data Sample")
    st.dataframe(df.head())

    # =============================
    # SCALING
    # =============================

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    seq_len = 24
    X, y = prepare_data(pd.DataFrame(scaled_data), seq_len)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # =============================
    # TRAIN MODEL
    # =============================

    with st.spinner("Training Transformer Weather AI..."):

        model = build_transformer_model(seq_len, df.shape[1])

        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=32,
            verbose=0
        )

    st.success("Model Training Completed")

    # =============================
    # FORECAST
    # =============================

    last_seq = X[-1].reshape(1, seq_len, df.shape[1])

    forecast_scaled = []
    temp_seq = last_seq.copy()

    for _ in range(24):

        pred_scaled = model.predict(temp_seq, verbose=0)[0][0]

        new_step = temp_seq[0, -1].copy()
        new_step[0] = pred_scaled

        forecast_scaled.append(new_step)

        temp_seq = np.append(
            temp_seq[:, 1:, :],
            [[new_step]],
            axis=1
        )

    forecast_scaled = np.array(forecast_scaled)

    forecast_real = scaler.inverse_transform(forecast_scaled)
    forecast_temp = forecast_real[:, 0]

    # =============================
    # FUTURE TIME
    # =============================

    last_time = df.index[-1]

    future_hours = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=24,
        freq="H"
    )

    forecast_clean = [round(float(x), 2) for x in forecast_temp]

    forecast_df = pd.DataFrame({
        "Time": future_hours,
        "Temperature (°C)": forecast_clean
    })

    # =============================
    # GRAPH
    # =============================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index[-48:],
        y=df["temp"].tail(48),
        name="Past Temperature"
    ))

    fig.add_trace(go.Scatter(
        x=future_hours,
        y=forecast_clean,
        name="Forecast Temperature"
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature °C"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # FORECAST TABLE
    # =============================

    st.subheader("🌡️ Next 24 Hour AI Temperature Forecast")
    st.dataframe(forecast_df, use_container_width=True)

    # =============================
    # SUMMARY METRICS
    # =============================

    col1, col2, col3 = st.columns(3)

    col1.metric("🌤️ Avg Temp", f"{np.mean(forecast_clean):.1f} °C")
    col2.metric("🔥 Max Temp", f"{np.max(forecast_clean):.1f} °C")
    col3.metric("❄️ Min Temp", f"{np.min(forecast_clean):.1f} °C")