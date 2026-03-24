"""
preprocessing.py
----------------
Load raw dataset, aggregate to daily sales, normalize, create sequences.
"""

import random, numpy as np, pandas as pd
random.seed(42)
np.random.seed(42)
import pickle
from sklearn.preprocessing import MinMaxScaler

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = r"dataset\retail_store_inventory.csv"
SEQ_LEN     = 14          # 14-day window to predict next day
TRAIN_RATIO = 0.80
SCALER_PATH = "scaler.pkl"


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    daily = (
        df.groupby("Date")["Units Sold"]
        .sum()
        .reset_index()
        .sort_values("Date")
        .rename(columns={"Units Sold": "sales"})
    )
    return daily


# ── 2. Validate ──────────────────────────────────────────────────────────────
def validate(daily: pd.DataFrame) -> None:
    assert daily.isnull().sum().sum() == 0, "Missing values found"
    assert (daily["sales"] >= 0).all(),     "Negative sales found"
    print(f"[OK] {len(daily)} daily records | "
          f"{daily['Date'].min().date()} → {daily['Date'].max().date()}")


# ── 3. Normalize ─────────────────────────────────────────────────────────────
def normalize(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    return scaled, scaler


# ── 4. Sequences ─────────────────────────────────────────────────────────────
def make_sequences(scaled: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len])
    return np.array(X), np.array(y)


# ── 5. Train/test split (time-based) ─────────────────────────────────────────
def split(X: np.ndarray, y: np.ndarray, ratio: float) -> tuple:
    cut = int(len(X) * ratio)
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    daily  = load_and_aggregate(DATA_PATH)
    validate(daily)

    values         = daily["sales"].values.astype(float)
    scaled, scaler = normalize(values)

    # Save scaler for inference
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] Scaler saved → {SCALER_PATH}")

    X, y = make_sequences(scaled, SEQ_LEN)
    X    = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, features)

    X_train, X_test, y_train, y_test = split(X, y, TRAIN_RATIO)
    print(f"[OK] Train: {X_train.shape} | Test: {X_test.shape}")

    np.save("X_train.npy", X_train)
    np.save("X_test.npy",  X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy",  y_test)
    print("[OK] Arrays saved → X_train/test, y_train/test .npy")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    run()