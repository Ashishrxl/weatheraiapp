import requests
import pandas as pd
from geopy.geocoders import Nominatim
from cache_manager import save_cache, load_cache, DATA_CACHE

def get_location(city):
    geo = Nominatim(user_agent="multi_weather_ai")
    loc = geo.geocode(city)
    if loc:
        return loc.latitude, loc.longitude
    return None, None


def fetch_weather(lat, lon, mode="forecast"):

    cache_file = f"{DATA_CACHE}/{mode}_{lat}_{lon}.pkl"
    cached = load_cache(cache_file)
    if cached is not None:
        return cached

    if mode == "forecast":
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"
    else:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-12-31&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation"

    data = requests.get(url).json()

    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temp": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind": data["hourly"]["wind_speed_10m"],
        "pressure": data["hourly"]["pressure_msl"],
        "rain": data["hourly"]["precipitation"]
    })

    df["time"] = pd.to_datetime(df["time"])

    save_cache(df, cache_file)
    return df