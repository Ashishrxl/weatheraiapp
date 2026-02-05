import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from cache_manager import MODEL_CACHE
import os

SEQ_LEN = 24
FEATURES = ["temp","humidity","wind","pressure","rain"]


def create_sequences(data):

    X = []
    y = []

    for i in range(len(data) - SEQ_LEN):
        X.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN])

    return np.array(X), np.array(y)


def build_model():

    inputs = layers.Input(shape=(SEQ_LEN, len(FEATURES)))

    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    outputs = layers.Dense(len(FEATURES))(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")

    return model


def train_multi_transformer(df, location_key):

    model_path = f"{MODEL_CACHE}/multi_{location_key}.h5"

    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)

    values = df[FEATURES].values

    X, y = create_sequences(values)

    model = build_model()
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    model.save(model_path)

    return model


def predict_multi(model, df):

    values = df[FEATURES].values

    X, _ = create_sequences(values)
    preds = model.predict(X)

    return preds