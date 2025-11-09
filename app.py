# ======================================================
# 🤖 AI STOCK PREDICTOR (FinBERT + XGBoost + Streamlit)
# ======================================================
# Created by Parth Khandelwal
# Features:
# - Stock Price Visualization (Candlestick + MA + Volume)
# - FinBERT News Sentiment Analysis
# - XGBoost Next-Day Prediction
# ======================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import datetime

# ---------------------------------------
# PAGE SETTINGS
# ---------------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")
st.title("🤖 AI Stock Predictor (FinBERT + XGBoost + Streamlit)")
st.markdown("An intelligent dashboard for **stock prediction** and **news sentiment analysis** using AI.")

# ---------------------------------------
# LOAD FINBERT MODEL (cached for speed)
# ---------------------------------------
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

# ---------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------
st.sidebar.header("⚙️ Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

chart_days = st.sidebar.selectbox(
    "Select chart display range:",
    options=[90, 180, 365, 730],
    format_func=lambda x: f"Last {x} Days"
)

# ---------------------------------------
# FETCH + CLEAN DATA
# ---------------------------------------
@st.cache_data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data = data.reset_index()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

# ---------------------------------------
# MAIN DASHBOARD
# ---------------------------------------
try:
    df = fetch_data(symbol, start_date, end_date)

    if df.empty:
        st.error("⚠️ No stock data found. Try a different s
