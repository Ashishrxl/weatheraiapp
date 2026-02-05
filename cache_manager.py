import os
import joblib

DATA_CACHE = "data_cache"
MODEL_CACHE = "model_cache"

os.makedirs(DATA_CACHE, exist_ok=True)
os.makedirs(MODEL_CACHE, exist_ok=True)


def save_cache(obj, path):
    joblib.dump(obj, path)


def load_cache(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None