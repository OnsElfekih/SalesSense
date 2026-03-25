"""
visualise_predictions.py
------------------------
Plot actual vs predicted sales on the test set.
Requires: y_true.npy, lstm_y_pred.npy, gru_y_pred.npy
Run AFTER compare.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Ensure photos folder exists ───────────────────────────────────────────────
if not os.path.exists("photos"):
    os.makedirs("photos")

# ── Load arrays ───────────────────────────────────────────────────────────────
FILES = ["y_true.npy", "lstm_y_pred.npy", "gru_y_pred.npy"]
if not all(os.path.exists(f) for f in FILES):
    print("[ERROR] Missing .npy files. Run compare.py first.")
    exit()

y_true    = np.load("y_true.npy")
lstm_pred = np.load("lstm_y_pred.npy")
gru_pred  = np.load("gru_y_pred.npy")

n = len(y_true)
x = np.arange(n)

# ── Metrics ───────────────────────────────────────────────────────────────────
lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_pred))
lstm_mae  = mean_absolute_error(y_true, lstm_pred)
gru_rmse  = np.sqrt(mean_squared_error(y_true, gru_pred))
gru_mae   = mean_absolute_error(y_true, gru_pred)

print(f"LSTM — RMSE: {lstm_rmse:.2f}  MAE: {lstm_mae:.2f}")
print(f"GRU  — RMSE: {gru_rmse:.2f}  MAE: {gru_mae:.2f}")

# ── Sanity check — predictions should not be flat ────────────────────────────
lstm_std = np.std(lstm_pred)
gru_std  = np.std(gru_pred)
actual_std = np.std(y_true)
print(f"\nActual std: {actual_std:.2f}  |  LSTM pred std: {lstm_std:.2f}  |  GRU pred std: {gru_std:.2f}")
if lstm_std < 100 or gru_std < 100:
    print("[WARNING] Predictions look flat — re-run compare.py to regenerate .npy files.")

# ── Plot 1: Full test set ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, y_true,    label="Actual",    color="#1F497D", linewidth=1.4, alpha=0.9)
ax.plot(x, lstm_pred, label=f"LSTM Pred  (RMSE={lstm_rmse:.0f})", 
        color="#ED7D31", linewidth=1.2, linestyle="--", alpha=0.85)
ax.plot(x, gru_pred,  label=f"GRU Pred   (RMSE={gru_rmse:.0f})", 
        color="#70AD47", linewidth=1.2, linestyle=":",  alpha=0.85)

ax.set_title("Actual vs Predicted — Test Set (144 days)", fontsize=13, fontweight="bold")
ax.set_xlabel("Test Day Index")
ax.set_ylabel("Units Sold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("photos/prediction_full.png", dpi=130, bbox_inches="tight")
print("\nSaved → photos/prediction_full.png")
plt.close()

# ── Plot 2: Zoomed — first 60 days (shows detail better) ─────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
z = 60
ax.plot(x[:z], y_true[:z],    label="Actual",    color="#1F497D", linewidth=1.6)
ax.plot(x[:z], lstm_pred[:z], label=f"LSTM Pred  (RMSE={lstm_rmse:.0f})",
        color="#ED7D31", linewidth=1.3, linestyle="--")
ax.plot(x[:z], gru_pred[:z],  label=f"GRU Pred   (RMSE={gru_rmse:.0f})",
        color="#70AD47", linewidth=1.3, linestyle=":")

ax.set_title("Actual vs Predicted — First 60 Test Days (zoomed)", fontsize=13, fontweight="bold")
ax.set_xlabel("Test Day Index")
ax.set_ylabel("Units Sold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("photos/prediction_zoom.png", dpi=130, bbox_inches="tight")
print("Saved → photos/prediction_zoom.png")
plt.close()

# ── Plot 3: Residuals (error over time) ───────────────────────────────────────
residuals_lstm = y_true - lstm_pred
residuals_gru  = y_true - gru_pred

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

axes[0].plot(x, residuals_lstm, color="#ED7D31", linewidth=0.9, alpha=0.8)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].fill_between(x, residuals_lstm, 0, alpha=0.15, color="#ED7D31")
axes[0].set_ylabel("Error (units)")
axes[0].set_title(f"LSTM Residuals  (MAE={lstm_mae:.0f})", fontsize=11)
axes[0].grid(axis="y", alpha=0.3)

axes[1].plot(x, residuals_gru, color="#70AD47", linewidth=0.9, alpha=0.8)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].fill_between(x, residuals_gru, 0, alpha=0.15, color="#70AD47")
axes[1].set_xlabel("Test Day Index")
axes[1].set_ylabel("Error (units)")
axes[1].set_title(f"GRU Residuals   (MAE={gru_mae:.0f})", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Prediction Residuals — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("photos/prediction_residuals.png", dpi=130, bbox_inches="tight")
print("Saved → photos/prediction_residuals.png")
plt.close()

print("\nAll charts saved in photos/ folder. Use photos/prediction_full.png in your report.")