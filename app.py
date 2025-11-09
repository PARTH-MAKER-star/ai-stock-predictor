import streamlit as st
from streamlit_tradingview_chart import tradingview_chart
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

st.title("📊 AI Stock Predictor + Live TradingView Chart")
st.markdown("Lightweight cloud version with XGBoost model and live TradingView data feed.")

# --------------------------
# SIDEBAR SETTINGS
# --------------------------
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol (e.g. NSE:RELIANCE, AAPL, TSLA)", "NSE:RELIANCE")
interval = st.sidebar.selectbox("Chart Timeframe", ["1", "5", "15", "60", "240", "D", "W", "M"], index=5)
theme = st.sidebar.radio("Theme", ["dark", "light"], horizontal=True)
auto_refresh = st.sidebar.checkbox("🔁 Auto-refresh every 1 minute", value=False)

# --------------------------
# TRADINGVIEW LIVE CHART
# --------------------------
st.subheader("📈 Live TradingView Chart")
tradingview_chart(
    symbol=symbol,
    interval=interval,
    theme=theme,
    autosize=True,
    height=600
)

# --------------------------
# FETCH HISTORICAL DATA
# --------------------------
st.subheader("📊 Historical Data + AI Direction Prediction")

ticker = symbol.split(":")[-1] + ".NS" if "NSE:" in symbol else symbol

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * 2)

try:
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    if df.empty:
        st.warning("⚠️ No data available for this symbol.")
    else:
        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        # Normalize data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
        y = np.where(df["Return"].shift(-1) > 0, 1, 0)

        # Train-Test Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # XGBoost Model
        model = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        # Display Results
        st.metric("📊 Model Accuracy", f"{acc*100:.2f}%")

        latest_data = X[-1].reshape(1, -1)
        prediction = model.predict(latest_data)[0]

        if prediction == 1:
            st.success("📈 AI Prediction: Stock likely to **rise tomorrow**.")
        else:
            st.error("📉 AI Prediction: Stock likely to **fall tomorrow**.")
except Exception as e:
    st.error(f"Error fetching data: {e}")

# --------------------------
# AUTO REFRESH
# --------------------------
if auto_refresh:
    import time
    st.toast("🔁 Auto-refreshing every 1 minute...")
    time.sleep(60)
    st.experimental_rerun()
Updated app.py (final working version)
