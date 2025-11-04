# ===========================================
# AI STOCK PREDICTOR APP
# ===========================================
# Built with Streamlit + FinBERT + XGBoost
# Predicts next-day stock movement using sentiment and price data
# ===========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import datetime
import requests

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")
st.title("🤖 AI Stock Predictor (FinBERT + XGBoost)")
st.markdown("Predict stock sentiment & next-day price movement using AI.")

# -------------------------------
# LOAD FINBERT MODEL
# -------------------------------
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_finbert()

# -------------------------------
# SENTIMENT ANALYSIS FUNCTION
# -------------------------------
def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = softmax(outputs.logits.detach().numpy()[0])
    sentiment = ["negative", "neutral", "positive"][np.argmax(scores)]
    return sentiment, scores

# -------------------------------
# STOCK DATA FETCH
# -------------------------------
@st.cache_data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# -------------------------------
# USER INPUT
# -------------------------------
st.sidebar.header("📊 Stock Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# -------------------------------
# LOAD STOCK DATA
# -------------------------------
try:
    df = fetch_data(symbol, start_date, end_date)
    st.subheader(f"Stock Data for {symbol}")
    st.dataframe(df.tail())

    # -------------------------------
    # PRICE CHART
    # -------------------------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))
    fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # SENTIMENT INPUT
    # -------------------------------
    st.subheader("📰 News Sentiment Analysis")
    news_text = st.text_area("Enter latest stock-related news or tweet:")

    if st.button("Analyze Sentiment"):
        if news_text.strip():
            sentiment, scores = get_finbert_sentiment(news_text)
            st.success(f"**Predicted Sentiment:** {sentiment.capitalize()}")
            st.json({
                "Negative": float(scores[0]),
                "Neutral": float(scores[1]),
                "Positive": float(scores[2])
            })
        else:
            st.warning("Please enter some news text first!")

    # -------------------------------
    # SIMPLE PREDICTION MODEL
    # -------------------------------
    st.subheader("📈 Predict Next-Day Movement")

    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
    y = np.where(df["Return"].shift(-1) > 0, 1, 0)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
    )
    model_xgb.fit(X_train, y_train)

    acc = model_xgb.score(X_test, y_test)
    st.metric("📊 Model Accuracy", f"{acc * 100:.2f}%")

    latest_data = X[-1].reshape(1, -1)
    pred = model_xgb.predict(latest_data)[0]
    if pred == 1:
        st.success("🔺 Prediction: The stock might **rise tomorrow.**")
    else:
        st.error("🔻 Prediction: The stock might **fall tomorrow.**")

except Exception as e:
    st.error(f"Error loading stock data: {e}")
