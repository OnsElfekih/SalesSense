"""
compare.py
----------
Full pipeline:
  1. Preprocess
  2. Train LSTM + GRU on full 80/20 split (saves weights + prediction arrays)
  3. 5-fold walk-forward CV with inverse-transformed RMSE (model selection)
  4. Select best model — fixed seed for reproducibility
  5. Save best_model.h5
"""

import os, sys, shutil, pickle, random
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fix seed BEFORE any TensorFlow import
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

sys.path.insert(0, os.path.dirname(__file__))

import preprocessing, train_lstm, train_gru

import tensorflow as tf
tf.random.set_seed(SEED)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

SEQ_LEN    = 14
EPOCHS     = 30
BATCH_SIZE = 32


def build_lstm():
    tf.random.set_seed(SEED)
    m = Sequential([Input(shape=(SEQ_LEN, 1)),
                    LSTM(64, return_sequences=True), Dropout(0.2),
                    LSTM(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer=Adam(1e-3), loss="mse")
    return m

def build_gru():
    tf.random.set_seed(SEED)
    m = Sequential([Input(shape=(SEQ_LEN, 1)),
                    GRU(64, return_sequences=True), Dropout(0.2),
                    GRU(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer=Adam(1e-3), loss="mse")
    return m

def inverse(arr, scaler):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def run_cv(X, y, scaler):
    tscv = TimeSeriesSplit(n_splits=5)
    lstm_rmses, gru_rmses = [], []
    lstm_maes,  gru_maes  = [], []

    print(f"\n{'='*62}")
    print("  Walk-Forward CV  (5 folds — metrics in original units)")
    print(f"{'='*62}\n")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        y_val_raw   = inverse(y_val, scaler)

        print(f"  Fold {fold}/5 — Train: {len(train_idx)} | Val: {len(val_idx)}")

        # LSTM
        lstm_m = build_lstm()
        lstm_m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   validation_split=0.1, verbose=0)
        lr = float(np.sqrt(mean_squared_error(
                y_val_raw, inverse(lstm_m.predict(X_val, verbose=0).flatten(), scaler))))
        lm = float(mean_absolute_error(
                y_val_raw, inverse(lstm_m.predict(X_val, verbose=0).flatten(), scaler)))
        lstm_rmses.append(lr); lstm_maes.append(lm)
        print(f"    LSTM → RMSE: {lr:.2f} units  MAE: {lm:.2f} units")

        # GRU
        gru_m = build_gru()
        gru_m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_split=0.1, verbose=0)
        gr = float(np.sqrt(mean_squared_error(
                y_val_raw, inverse(gru_m.predict(X_val, verbose=0).flatten(), scaler))))
        gm = float(mean_absolute_error(
                y_val_raw, inverse(gru_m.predict(X_val, verbose=0).flatten(), scaler)))
        gru_rmses.append(gr); gru_maes.append(gm)
        print(f"    GRU  → RMSE: {gr:.2f} units  MAE: {gm:.2f} units\n")

    print(f"{'='*62}")
    print("  CV SUMMARY  (original units)")
    print(f"{'='*62}")
    print(f"  {'Model':<6} {'Mean RMSE':>11} {'Std':>8} {'Mean MAE':>11} {'Std':>8}")
    print(f"  {'-'*48}")
    print(f"  {'LSTM':<6} {np.mean(lstm_rmses):>11.2f} {np.std(lstm_rmses):>8.2f} "
          f"{np.mean(lstm_maes):>11.2f} {np.std(lstm_maes):>8.2f}")
    print(f"  {'GRU':<6} {np.mean(gru_rmses):>11.2f} {np.std(gru_rmses):>8.2f} "
          f"{np.mean(gru_maes):>11.2f} {np.std(gru_maes):>8.2f}")
    print(f"{'='*62}")

    best = "GRU" if np.mean(gru_rmses) <= np.mean(lstm_rmses) else "LSTM"
    return best, np.mean(lstm_rmses), np.mean(gru_rmses)


def main():
    # 1. Preprocess
    print("\n" + "="*50)
    print("  STEP 1: Preprocessing")
    print("="*50)
    preprocessing.run()

    # 2. Train both on full 80/20 split (saves .h5 + .npy files)
    print("\n" + "="*50)
    print("  STEP 2: Train LSTM  (80/20 — saves weights)")
    print("="*50)
    lstm_rmse, lstm_mae = train_lstm.run()

    print("\n" + "="*50)
    print("  STEP 3: Train GRU   (80/20 — saves weights)")
    print("="*50)
    gru_rmse, gru_mae = train_gru.run()

    # 3. Single-split summary (informational)
    print(f"\n{'='*50}")
    print("  SINGLE-SPLIT  (80/20) — informational only")
    print(f"{'='*50}")
    print(f"  {'Model':<10} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*30}")
    print(f"  {'LSTM':<10} {lstm_rmse:>10.2f} {lstm_mae:>10.2f}")
    print(f"  {'GRU':<10} {gru_rmse:>10.2f} {gru_mae:>10.2f}")
    print(f"  Note: varies with random init — not used for selection.")
    print(f"{'='*50}")

    # 4. CV with inverse-transformed metrics
    X = np.concatenate([np.load("X_train.npy"), np.load("X_test.npy")])
    y = np.concatenate([np.load("y_train.npy"), np.load("y_test.npy")])
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    best, lstm_cv, gru_cv = run_cv(X, y, scaler)

    # 5. Save best model
    src = "gru_model.h5" if best == "GRU" else "lstm_model.h5"
    shutil.copy(src, "best_model.h5")

    print(f"\n{'='*50}")
    print(f"  [SELECTED]  Best model : {best}  (5-fold mean CV RMSE)")
    print(f"  LSTM mean CV RMSE : {lstm_cv:.2f} units")
    print(f"  GRU  mean CV RMSE : {gru_cv:.2f} units")
    print(f"  Copied {src} → best_model.h5")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()