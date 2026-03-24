"""
streamlit_app.py
----------------
Sales Forecasting Dashboard — LSTM / GRU
Run: streamlit run streamlit_app.py
"""

import os, pickle, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

# ── Resolve paths — works whether CSV is in root or dataset/ subfolder ────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    """Search root folder and common subfolders."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "dataset", filename),
        os.path.join(BASE_DIR, "data",    filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, filename)   # fallback — will raise clear error

# ── Load model + scaler (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model(find_file("best_model.h5"), compile=False)
    with open(find_file("scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_test_data():
    X_test  = np.load(find_file("X_test.npy"))
    y_true  = np.load(find_file("y_true.npy"))
    lstm_p  = np.load(find_file("lstm_y_pred.npy"))
    gru_p   = np.load(find_file("gru_y_pred.npy"))
    return X_test, y_true, lstm_p, gru_p

@st.cache_data
def load_daily():
    df = pd.read_csv(find_file("retail_store_inventory.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    daily = (df.groupby("Date")["Units Sold"].sum()
               .reset_index().sort_values("Date")
               .rename(columns={"Units Sold": "sales"}))
    return daily

model, scaler = load_artifacts()
X_test, y_true, lstm_pred, gru_pred = load_test_data()
daily = load_daily()

SEQ_LEN = 14

# ── Metrics ───────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error
lstm_rmse = float(np.sqrt(mean_squared_error(y_true, lstm_pred)))
lstm_mae  = float(mean_absolute_error(y_true, lstm_pred))
gru_rmse  = float(np.sqrt(mean_squared_error(y_true, gru_pred)))
gru_mae   = float(mean_absolute_error(y_true, gru_pred))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Navigation")
    page = st.radio("Go to", ["🏠 Overview", "📊 EDA", "🤖 Model Results", "🔮 Predict"])
    st.divider()
    st.caption("Dataset: Retail Store Inventory")
    st.caption("Period: 2022-01-01 → 2024-01-01")
    st.caption("Records: 73,100  |  Days: 731")
    st.caption("Best model: GRU  (5-fold CV)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📈 Retail Sales Forecasting")
    st.markdown("**Deep Learning time-series prediction using LSTM and GRU networks.**")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", "GRU")
    c2.metric("GRU CV RMSE",  "1,041.34 units")
    c3.metric("GRU Test MAE", f"{gru_mae:,.0f} units")
    c4.metric("Test Samples", "144 days")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Pipeline")
        st.markdown("""
1. **Preprocessing** — aggregate daily sales, normalise, 14-day windows
2. **Train/Test Split** — 80% train / 20% test, time-ordered
3. **LSTM** — 2 layers (64→32), Dropout 0.2, Dense output
4. **GRU** — same structure, 24% fewer parameters
5. **5-Fold CV** — walk-forward TimeSeriesSplit → GRU selected
6. **Deployment** — this Streamlit dashboard
        """)

    with col2:
        st.subheader("Model Comparison")
        df_cmp = pd.DataFrame({
            "Model":        ["LSTM", "GRU"],
            "Test RMSE":    [round(lstm_rmse, 2), round(gru_rmse, 2)],
            "Test MAE":     [round(lstm_mae,  2), round(gru_mae,  2)],
            "CV RMSE":      [1045.84, 1041.34],
            "Parameters":   [29345, 22305],
            "Selected":     ["❌", "✅"],
        })
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Time Series", "Monthly Pattern", "Distribution"])

    with tab1:
        st.subheader("Daily Total Units Sold — 2022 to 2024")
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(daily["Date"], daily["sales"], linewidth=0.9, color="#1F497D")
        ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption("Weekly oscillations visible throughout. Mild Q4 uplift in both years.")

    with tab2:
        st.subheader("Monthly Average Daily Sales")
        daily["Month"] = daily["Date"].dt.to_period("M").astype(str)
        monthly = daily.groupby("Month")["sales"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.bar(range(len(monthly)), monthly["sales"],
               color="#2E75B6", edgecolor="white", linewidth=0.3)
        tick_idx = list(range(0, len(monthly), 3))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([monthly["Month"].iloc[i] for i in tick_idx], rotation=30)
        ax.set_ylabel("Avg Units Sold"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.subheader("Sales Distribution")
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.hist(daily["sales"], bins=40, color="#2E75B6", edgecolor="white", linewidth=0.4)
        ax.axvline(daily["sales"].mean(), color="#ED7D31", linewidth=2,
                   label=f"Mean: {daily['sales'].mean():,.0f}")
        ax.set_xlabel("Daily Units Sold"); ax.set_ylabel("Frequency")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean",  f"{daily['sales'].mean():,.0f}")
        c2.metric("Std",   f"{daily['sales'].std():,.0f}")
        c3.metric("Min",   f"{daily['sales'].min():,.0f}")
        c4.metric("Max",   f"{daily['sales'].max():,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LSTM RMSE", f"{lstm_rmse:,.2f}")
    c2.metric("LSTM MAE",  f"{lstm_mae:,.2f}")
    c3.metric("GRU RMSE",  f"{gru_rmse:,.2f}",
              delta=f"{gru_rmse - lstm_rmse:+.2f} vs LSTM", delta_color="inverse")
    c4.metric("GRU MAE",   f"{gru_mae:,.2f}",
              delta=f"{gru_mae - lstm_mae:+.2f} vs LSTM",  delta_color="inverse")

    st.divider()
    tab1, tab2 = st.tabs(["Predictions", "Cross-Validation"])

    with tab1:
        st.subheader("Actual vs Predicted — Test Set (144 days)")
        n  = len(y_true)
        xr = np.arange(n)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6))

        # Raw predictions
        axes[0].plot(xr, y_true,    label="Actual",
                     color="#1F497D", linewidth=1.1, alpha=0.85)
        axes[0].plot(xr, lstm_pred, label=f"LSTM (RMSE={lstm_rmse:.0f})",
                     color="#ED7D31", linewidth=1.4, linestyle="--")
        axes[0].plot(xr, gru_pred,  label=f"GRU  (RMSE={gru_rmse:.0f})",
                     color="#70AD47", linewidth=1.4, linestyle=":")
        axes[0].set_ylabel("Units Sold"); axes[0].legend(fontsize=9)
        axes[0].set_title("Raw daily predictions"); axes[0].grid(axis="y", alpha=0.3)

        # 7-day rolling average
        def rolling(arr, w=7):
            return np.convolve(arr, np.ones(w)/w, mode="valid")
        xrr = np.arange(len(rolling(y_true)))
        axes[1].plot(xrr, rolling(y_true),    label="Actual (7-day avg)",
                     color="#1F497D", linewidth=1.6)
        axes[1].plot(xrr, rolling(lstm_pred), label="LSTM  (7-day avg)",
                     color="#ED7D31", linewidth=1.4, linestyle="--")
        axes[1].plot(xrr, rolling(gru_pred),  label="GRU   (7-day avg)",
                     color="#70AD47", linewidth=1.4, linestyle=":")
        axes[1].set_xlabel("Test Day Index"); axes[1].set_ylabel("Units Sold")
        axes[1].set_title("7-day rolling average — trend tracking")
        axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.subheader("5-Fold Walk-Forward Cross-Validation")
        cv_data = {
            "Fold":      ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "Std"],
            "LSTM RMSE": [1025.27,  1019.71,  1089.26,  1064.19,  1030.77, 1045.84, 26.66],
            "GRU RMSE":  [1025.23,  1015.22,  1071.09,  1071.11,  1024.06, 1041.34, 24.54],
        }
        st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 3.5))
        x_pos = np.arange(5)
        ax.plot(x_pos, cv_data["LSTM RMSE"][:5], marker="o",
                label="LSTM", linewidth=2, color="#1F497D")
        ax.plot(x_pos, cv_data["GRU RMSE"][:5],  marker="s",
                label="GRU",  linewidth=2, color="#ED7D31")
        ax.axhline(1045.84, color="#1F497D", linestyle="--", alpha=0.5,
                   label="LSTM avg: 1045.84")
        ax.axhline(1041.34, color="#ED7D31", linestyle="--", alpha=0.5,
                   label="GRU avg:  1041.34")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5"])
        ax.set_ylabel("RMSE (units sold)")
        ax.set_title("RMSE across folds — GRU selected (lower mean CV RMSE)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.success("✅ GRU selected — Mean CV RMSE: 1,041.34 vs LSTM: 1,045.84  |  24% fewer parameters")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Predict Next-Day Sales")
    st.markdown("Enter the last **14 days** of daily sales to get tomorrow's forecast.")
    st.divider()

    tab_manual, tab_auto = st.tabs(["✏️ Manual Input", "🎲 Use Test Data"])

    with tab_manual:
        st.subheader("Enter 14 daily sales values")
        cols = st.columns(7)
        defaults = [13000,13500,12000,14000,15000,13200,12800,
                    13700,14100,13900,12500,13000,14500,13300]
        vals = []
        for i in range(14):
            v = cols[i % 7].number_input(
                f"Day {i+1}", min_value=0, max_value=100000,
                value=defaults[i], step=100, key=f"d{i}")
            vals.append(v)

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            arr    = np.array(vals, dtype=float)
            scaled = scaler.transform(arr.reshape(-1, 1)).flatten()
            X      = scaled.reshape(1, SEQ_LEN, 1)
            pred_s = model.predict(X, verbose=0)[0][0]
            pred   = float(scaler.inverse_transform([[pred_s]])[0][0])

            st.success(f"### Predicted next-day sales: **{pred:,.0f} units**")

            fig, ax = plt.subplots(figsize=(9, 3))
            ax.bar(range(14), vals, color="#2E75B6", label="Input (14 days)")
            ax.bar(14, pred, color="#ED7D31", label=f"Prediction: {pred:,.0f}")
            ax.set_xticks(range(15))
            ax.set_xticklabels([f"D{i+1}" for i in range(14)] + ["D15\n(pred)"])
            ax.set_xlabel("Day"); ax.set_ylabel("Units Sold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab_auto:
        st.subheader("Pick a sample from the test set")
        idx = st.slider("Test sample index", 0, len(X_test) - 1, 0)

        sample_raw  = scaler.inverse_transform(
                          X_test[idx].reshape(-1, 1)).flatten()
        actual_next = float(y_true[idx])

        pred_s = model.predict(X_test[idx:idx+1], verbose=0)[0][0]
        pred   = float(scaler.inverse_transform([[pred_s]])[0][0])
        error  = abs(pred - actual_next)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", f"{pred:,.0f} units")
        c2.metric("Actual",    f"{actual_next:,.0f} units")
        c3.metric("Error",     f"{error:,.0f} units",
                  delta=f"{error/actual_next*100:.1f}%", delta_color="inverse")

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(range(14), sample_raw, color="#2E75B6", label="Input sequence")
        ax.bar(14, pred,         color="#ED7D31",
               label=f"Predicted: {pred:,.0f}")
        ax.bar(14, actual_next,  color="#70AD47", alpha=0.6, width=0.4,
               label=f"Actual: {actual_next:,.0f}")
        ax.set_xticks(range(15))
        ax.set_xticklabels([f"D{i+1}" for i in range(14)] + ["D15"])
        ax.set_xlabel("Day"); ax.set_ylabel("Units Sold")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()