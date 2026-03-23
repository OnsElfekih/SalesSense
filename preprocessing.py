
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = r"dataset\retail_store_inventory.csv" 
SEQ_LEN     = 14          # number of days in input sequence
TRAIN_RATIO = 0.80
SCALER_PATH = "scaler.pkl"  # file to save the scaler


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])  # # convert Date column to datetime
    daily = (
        df.groupby("Date")["Units Sold"]  # group by date
        .sum()                             # sum sales per day
        .reset_index()                     # reset index
        .sort_values("Date")               # sort by date ascending
        .rename(columns={"Units Sold": "sales"})  # rename column to 'sales'
    )
    return daily


# ── 2. Validate ──────────────────────────────────────────────────────────────
def validate(daily: pd.DataFrame) -> None:
    assert daily.isnull().sum().sum() == 0, "Missing values found"
    assert (daily["sales"] >= 0).all(),     "Negative sales found"
    print(f"[OK] {len(daily)} daily records | "
          f"{daily['Date'].min().date()} → {daily['Date'].max().date()}") # print number of days and date range


# ── 3. Normalize ─────────────────────────────────────────────────────────────
def normalize(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    return scaled, scaler  # return normalized values and the scaler


# ── 4. Sequences ─────────────────────────────────────────────────────────────
def make_sequences(scaled: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []  # lists to store input sequences and targets
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len])
    return np.array(X), np.array(y)  # convert lists to numpy arrays
 

# ── 5. Train/test split (time-based) ─────────────────────────────────────────
def split(X: np.ndarray, y: np.ndarray, ratio: float) -> tuple:
    cut = int(len(X) * ratio)
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    daily  = load_and_aggregate(DATA_PATH)
    validate(daily)

    values         = daily["sales"].values.astype(float)  # extract sales as float
    scaled, scaler = normalize(values)                   # normalize sales values

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