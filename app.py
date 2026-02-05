import streamlit as st
import matplotlib.pyplot as plt

from weather_api import get_location, fetch_weather
from multi_transformer import train_multi_transformer, predict_multi, FEATURES

st.title("🌍 Multi-Variable Transformer Weather AI")

city = st.text_input("Enter City")

if city:

    lat, lon = get_location(city)

    if lat is None:
        st.error("City not found")
        st.stop()

    st.success(f"Location: {city}")

    hist_df = fetch_weather(lat, lon, "history")
    forecast_df = fetch_weather(lat, lon, "forecast")

    location_key = f"{lat}_{lon}"

    model = train_multi_transformer(hist_df, location_key)

    preds = predict_multi(model, forecast_df)

    forecast_df = forecast_df.iloc[24:].copy()

    for i, col in enumerate(FEATURES):
        forecast_df[f"AI_{col}"] = preds[:,i]

    st.subheader("Temperature Forecast")

    fig, ax = plt.subplots()
    ax.plot(forecast_df["time"], forecast_df["temp"], label="API Temp")
    ax.plot(forecast_df["time"], forecast_df["AI_temp"], label="AI Temp")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Rain Prediction")

    fig2, ax2 = plt.subplots()
    ax2.plot(forecast_df["time"], forecast_df["rain"], label="API Rain")
    ax2.plot(forecast_df["time"], forecast_df["AI_rain"], label="AI Rain")
    ax2.legend()
    st.pyplot(fig2)

    st.dataframe(forecast_df.head(48))