
# STOCK INVESTMENT FORECASTER (STREAMLIT + YFINANCE)

import importlib
import importlib.util
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Optional: Use scikit-learn if available
SklearnLinearRegression = None
SklearnRandomForestRegressor = None
sklearn_mae = None
sklearn_mse = None
sklearn_r2 = None
sklearn_search = None

if importlib.util.find_spec("sklearn") is not None:
    try:
        sklearn_linear_model = importlib.import_module("sklearn.linear_model")
        sklearn_ensemble = importlib.import_module("sklearn.ensemble")
        sklearn_metrics = importlib.import_module("sklearn.metrics")
        sklearn_model_selection = importlib.import_module("sklearn.model_selection")

        SklearnLinearRegression = getattr(sklearn_linear_model, "LinearRegression", None)
        SklearnRandomForestRegressor = getattr(sklearn_ensemble, "RandomForestRegressor", None)
        sklearn_mae = getattr(sklearn_metrics, "mean_absolute_error", None)
        sklearn_mse = getattr(sklearn_metrics, "mean_squared_error", None)
        sklearn_r2 = getattr(sklearn_metrics, "r2_score", None)
        sklearn_search = getattr(sklearn_model_selection, "RandomizedSearchCV", None)

    except Exception:
        pass

SKLEARN_AVAILABLE = all([
    SklearnLinearRegression,
    sklearn_mae,
    sklearn_mse,
    sklearn_r2
])

# Fallback Linear Regression (if sklearn missing)
def _ensure_2d(features):
    arr = np.asarray(features, dtype="float64")
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

class SimpleLinearRegression:
    def __init__(self):
        self._coef = None
        self._intercept = None

    def fit(self, X, y):
        X = _ensure_2d(X)
        Xb = np.c_[np.ones(X.shape[0]), X]
        beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self._intercept = beta[0]
        self._coef = beta[1:]

    def predict(self, X):
        X = _ensure_2d(X)
        return self._intercept + X @ self._coef

# Select correct model backend
LinearRegression = SklearnLinearRegression if SKLEARN_AVAILABLE else SimpleLinearRegression
RandomForestRegressor = SklearnRandomForestRegressor

# Select metric functions
mean_absolute_error = sklearn_mae if SKLEARN_AVAILABLE else lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
mean_squared_error = sklearn_mse if SKLEARN_AVAILABLE else lambda y, p: float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))
r2_score = sklearn_r2 if SKLEARN_AVAILABLE else lambda y, p: 0.0

# STREAMLIT UI
st.set_page_config(page_title="Stock Investment Forecaster", page_icon="📈", layout="wide")

stocks = {
    "India ": {
        "RELIANCE.NS": "Reliance",
        "TCS.NS": "TCS",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "SBIN.NS": "SBI Bank",
    },
    "Global ": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "TSLA": "Tesla",
        "GOOGL": "Alphabet",
        "AMZN": "Amazon",
    },
}

st.sidebar.title("⚙️ Settings")
group = st.sidebar.selectbox("Select Market", list(stocks.keys()))
ticker = st.sidebar.selectbox("Choose Stock", list(stocks[group].keys()), format_func=lambda x: stocks[group][x])
years = st.sidebar.slider("Years of Historical Data", 3, 15, 7)

model_type = st.sidebar.selectbox("Model", ["Linear Regression", "Random Forest"])
enable_tuning = st.sidebar.checkbox("Tune Hyperparameters (Slow, but more accurate)")

invest_amount = st.sidebar.number_input("Your Investment Amount (₹ or $)", min_value=1000.0, value=10000.0, step=1000.0)

# DATA DOWNLOAD
start = date.today() - timedelta(days=years * 365)
data = yf.download(ticker, start=start, end=date.today(), auto_adjust=False)

st.title("STOCKZZZ")
st.write(f"Using data for **{stocks[group][ticker]}** ({ticker})")

if data.empty:
    st.error("No data available.")
    st.stop()

# FEATURE ENGINEERING (Drop NA only once)
data["Target"] = data["Close"].shift(-1)
data.dropna(inplace=True)

X = data[["Close"]]
y = data["Target"]

# TRAIN / TEST SPLIT
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# MODEL TRAINING & HYPERPARAMETER TUNING
if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
else:
    if enable_tuning and sklearn_search:
        param_grid = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 4, 6, 8, 10],
            "min_samples_split": [2, 5, 10],
        }
        search = sklearn_search(RandomForestRegressor(), param_grid, cv=3, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

pred = model.predict(X_test)

# METRICS
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

# INVESTMENT RETURN ESTIMATE
last_close = float(data["Close"].iloc[-1])
next_price = float(model.predict(_ensure_2d([last_close]))[0])
growth = (next_price / last_close) - 1
future_value = invest_amount * (1 + growth)

st.subheader("💰 Investment Projection")
st.write(f"Current Price: **{last_close:.2f}**")
st.write(f"Predicted Next Price: **{next_price:.2f}**")
st.write(f"Expected Change: **{growth*100:.2f}%**")
st.success(f"Estimated Future Value: **{future_value:.2f}**")

# CANDLESTICK CHART (Now Fixed ✅)
st.subheader("Candlestick Price Chart")
candlestick = go.Figure(data=[
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    )
])
candlestick.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False)
st.plotly_chart(candlestick, use_container_width=True)

# Disclaimer & Interpretation
with st.expander("Disclaimer"):
    st.write("Educational use only. Not financial advice.")

with st.expander("Inference (Model Quality Summary)"):
    if r2 >= 0.90:
        st.write("**R² Score:** Excellent — Trend understanding is strong ✅")
    elif r2 >= 0.75:
        st.write("**R² Score:** Good — Model captures trends reliably 👍")
    elif r2 >= 0.50:
        st.write("**R² Score:** Moderate — Predictions may vary 😐")
    else:
        st.write("**R² Score:** Poor — Trend not captured ⚠️")

    if mae <= 15:
        st.write("**MAE:** Good — Low average error ✅")
    elif mae <= 30:
        st.write("**MAE:** Acceptable — Medium error 😐")
    else:
        st.write("**MAE:** High — Predictions vary widely ⚠️")

    if rmse <= 25:
        st.write("**RMSE:** Good — Large mistakes are limited ✅")
    else:
        st.write("**RMSE:** High — Large errors occur ⚠️")

    st.write("---")
    if r2 >= 0.9 and mae < 20 and rmse < 25:
        st.success("✅ Overall: Model is performing **very well**.")
    elif r2 >= 0.75:
        st.info("ℹ Overall: Model performance is **acceptable**.")
    else:
        st.warning("⚠ Overall: Model is weak, consider adding technical signals.")
