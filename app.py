import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import gzip
import pickle
import concurrent.futures

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

# =========================
# CONFIG
# =========================

YEARS_OF_HISTORY = 1
CHUNK_DAYS = 30
CACHE_FILE = "weather_dataset.pkl.gz"

DEVICE = torch.device("cpu")

st.set_page_config(page_title="Weather AI", layout="wide")

# =========================
# LOCATION SEARCH
# =========================

@st.cache_data(ttl=86400)
def search_location(query):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 5}

    r = requests.get(url, params=params).json()
    return r.get("results", [])


# =========================
# WEATHER FETCH
# =========================

def fetch_historical_weather(lat, lon, start, end):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": [
            "temperature_2m",
            "relativehumidity_2m",
            "surface_pressure",
            "windspeed_10m"
        ],
        "timezone": "auto"
    }

    r = requests.get(url, params=params).json()

    if "hourly" not in r:
        return pd.DataFrame()

    df = pd.DataFrame(r["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df


def download_chunk(lat, lon, start, end):
    try:
        return fetch_historical_weather(lat, lon, start, end)
    except:
        return pd.DataFrame()


# =========================
# MULTI YEAR DATASET PIPELINE
# =========================

@st.cache_data(ttl=86400)
def load_or_build_dataset(lat, lon):

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=365 * YEARS_OF_HISTORY)

    df = pd.DataFrame()

    # =============================
    # SAFE CACHE LOAD
    # =============================
    if os.path.exists(CACHE_FILE):

        try:
            with gzip.open(CACHE_FILE, "rb") as f:
                df = pickle.load(f)

            df["time"] = pd.to_datetime(df["time"])

        except Exception:
            # ⭐ Corrupted cache auto-fix
            os.remove(CACHE_FILE)
            st.warning("Cache corrupted → rebuilding dataset")

            df = pd.DataFrame()

    # =============================
    # Detect missing ranges
    # =============================
    if len(df) > 0:
        oldest = df["time"].min().date()
        newest = df["time"].max().date()
    else:
        oldest = today
        newest = start_date

    missing_ranges = []

    if oldest > start_date:
        missing_ranges.append((start_date, oldest - timedelta(days=1)))

    if newest < today:
        missing_ranges.append((newest + timedelta(days=1), today))

    chunk_tasks = []

    for r_start, r_end in missing_ranges:
        current = r_start

        while current <= r_end:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), r_end)

            chunk_tasks.append((
                lat,
                lon,
                current.isoformat(),
                chunk_end.isoformat()
            ))

            current = chunk_end + timedelta(days=1)

    # =============================
    # Parallel Download
    # =============================
    if chunk_tasks:

        st.info(f"Downloading {len(chunk_tasks)} historical chunks...")

        parts = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_chunk, *task) for task in chunk_tasks]

            for f in concurrent.futures.as_completed(futures):
                data = f.result()

                if len(data) > 0:
                    parts.append(data)

        if parts:
            new_df = pd.concat(parts)
            df = pd.concat([df, new_df])

    if len(df) == 0:
        return df

    df = df[df["time"] <= pd.Timestamp.now()]
    df = df.drop_duplicates(subset="time")
    df = df.sort_values("time")

    # =============================
    # SAFE CACHE SAVE
    # =============================
    try:
        with gzip.open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
    except:
        st.warning("Cache save failed, continuing without cache")

    return df


# =========================
# TRANSFORMER MODEL
# =========================

class WeatherTransformer(nn.Module):

    def __init__(self, features=4, d_model=64):
        super().__init__()

        self.input_proj = nn.Linear(features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output(x[:, -1])
        return x


# =========================
# TRAIN MODEL
# =========================

def train_model(df):

    df = df.set_index("time")

    features = [
        "temperature_2m",
        "relativehumidity_2m",
        "surface_pressure",
        "windspeed_10m"
    ]

    df = df[features].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    SEQ = 48

    X, y = [], []

    for i in range(len(scaled) - SEQ):
        X.append(scaled[i:i+SEQ])
        y.append(scaled[i+SEQ][0])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = WeatherTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(5):
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, scaler, df


# =========================
# FORECAST
# =========================

def forecast(model, scaler, df):

    SEQ = 48
    HOURS = 24

    data = scaler.transform(df)[-SEQ:]
    seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    preds = []

    model.eval()

    for _ in range(HOURS):
        with torch.no_grad():
            p = model(seq).item()

        last = seq[0, -1].numpy()
        new_row = last.copy()
        new_row[0] = p

        preds.append(p)

        new_scaled = np.vstack([seq[0, 1:], new_row])
        seq = torch.tensor(new_scaled, dtype=torch.float32).unsqueeze(0)

    preds = np.array(preds)
    dummy = np.zeros((len(preds), scaler.scale_.shape[0]))
    dummy[:, 0] = preds

    preds = scaler.inverse_transform(dummy)[:, 0]

    return preds


# =========================
# STREAMLIT UI
# =========================

st.title("🌦 Multi-Year Transformer Weather AI")

query = st.text_input("Enter City Name")

if query:

    results = search_location(query)

    options = [
        f"{r['name']}, {r.get('country','')} ({r['latitude']}, {r['longitude']})"
        for r in results
    ]

    choice = st.selectbox("Select Location", options)

    idx = options.index(choice)
    lat = results[idx]["latitude"]
    lon = results[idx]["longitude"]

    st.write(f"Latitude: {lat}, Longitude: {lon}")

    # Load dataset
    df = load_or_build_dataset(lat, lon)

    if len(df) == 0:
        st.error("No data found")
        st.stop()

    st.success(f"Dataset rows: {len(df)}")
    st.info(f"Range: {df['time'].min()} → {df['time'].max()}")

    st.subheader("Training Data Sample")
    st.dataframe(df.tail(24))

    # Train
    model, scaler, clean_df = train_model(df)

    # Forecast
    preds = forecast(model, scaler, clean_df)

    future_times = [
        pd.Timestamp.now() + timedelta(hours=i+1)
        for i in range(len(preds))
    ]

    forecast_df = pd.DataFrame({
        "Time": future_times,
        "Predicted Temperature": preds
    })

    st.subheader("Next 24 Hour Forecast")
    st.dataframe(forecast_df)

    st.line_chart(forecast_df.set_index("Time"))