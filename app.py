# ======================================================
# 📈 AI STOCK PREDICTOR with LIVE TRADINGVIEW CHART
# ======================================================

import streamlit as st
from streamlit_tradingview_chart import tradingview_chart
import pandas as pd
import numpy as np
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import plotly.express as px

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

st.title("🤖 AI Stock Predictor + Live TradingView Chart")
st.markdown("An intelligent trading dashboard combining **TradingView**, **FinBERT sentiment**, and **XGBoost predictions**.")

# ---------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------
st.sidebar.header("⚙️ Chart & Settings")

symbol = st.sidebar.text_input("Stock Symbol", "NSE:RELIANCE")
interval = st.sidebar.selectbox("Timeframe", ["1", "5", "15", "60", "240", "D", "W", "M"], index=5)
theme = st.sidebar.radio("Theme", ["dark", "light"], horizontal=True)
auto_refresh = st.sidebar.checkbox("🔁 Auto-refresh every 1 min", value=False)

# ---------------------------------------
# 📊 TRADINGVIEW LIVE CHART
# ---------------------------------------
st.subheader("📊 Live Chart (TradingView)")
st.info("Fully interactive — zoom, pan, change timeframes, and view live data directly from TradingView.")

tradingview_chart(
    symbol=symbol,
    interval=interval,
    theme=theme,
    autosize=True,
    height=600
)

# ---------------------------------------
# 🧠 FinBERT SENTIMENT ANALYSIS
# ---------------------------------------
st.subheader("🧠 News Sentiment Analysis")

@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, finbert_model = load_finbert()

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    scores = softmax(outputs.logits.detach().numpy()[0])
    sentiment = ["Negative", "Neutral", "Positive"][np.argmax(scores)]
    return sentiment, scores

news_text = st.text_area("Paste recent news or tweet:")
if st.button("Analyze Sentiment"):
    if news_text.strip():
        sentiment, scores = get_finbert_sentiment(news_text)
        st.success(f"Predicted Sentiment: **{sentiment}**")
        df_sent = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Score": scores
        })
        fig_sent = px.bar(df_sent, x="Sentiment", y="Score", color="Sentiment",
                          title="FinBERT Sentiment Scores", text="Score", template="plotly_dark" if theme == "dark" else "plotly_white")
        fig_sent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.warning("Please enter text to analyze.")

# ---------------------------------------
# 🤖 XGBOOST PREDICTION
# ---------------------------------------
st.subheader("📈 Predict Next-Day Direction (AI Model)")

start_date = datetime.date.today() - datetime.timedelta(days=365 * 2)
end_date = datetime.date.today()

df = yf.download(symbol.split(":")[-1] + ".NS", start=start_date, end=end_date, interval="1d", progress=False)
if df.empty:
    st.warning("⚠️ No historical data available for this symbol.")
else:
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
    y = np.where(df["Return"].shift(-1) > 0, 1, 0)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    st.metric("📊 Model Accuracy", f"{acc*100:.2f}%")

    latest_data = X[-1].reshape(1, -1)
    prediction = model.predict(latest_data)[0]
    if prediction == 1:
        st.success("📈 AI Prediction: Stock likely to **rise tomorrow**.")
    else:
        st.error("📉 AI Prediction: Stock likely to **fall tomorrow**.")

# ---------------------------------------
# 🔄 AUTO REFRESH
# ---------------------------------------
if auto_refresh:
    import time
    st.toast("🔁 Auto-refreshing every 1 minute…")
    time.sleep(60)
    st.experimental_rerun()
