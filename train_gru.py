"""
train_gru.py
------------
Build, train, evaluate, and save GRU model.
Same architecture as LSTM — only the recurrent layer changes.
"""

import os, numpy as np, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

SEQ_LEN     = 14
EPOCHS      = 50
BATCH_SIZE  = 32
MODEL_PATH  = "gru_model.h5"
SCALER_PATH = "scaler.pkl"


def build_gru(seq_len: int) -> Sequential:
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary()
    return model


def evaluate(model, X_test, y_test, scaler):
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return rmse, mae, y_pred, y_true


def run():
    X_train = np.load("X_train.npy")
    X_test  = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test  = np.load("y_test.npy")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = build_gru(SEQ_LEN)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"[OK] Model saved → {MODEL_PATH}")

    rmse, mae, y_pred, y_true = evaluate(model, X_test, y_test, scaler)
    print(f"\n{'='*40}")
    print(f"  GRU Results")
    print(f"  RMSE : {rmse:.2f} units")
    print(f"  MAE  : {mae:.2f} units")
    print(f"{'='*40}\n")

    np.save("gru_y_pred.npy", y_pred)

    return rmse, mae


if __name__ == "__main__":
    run()