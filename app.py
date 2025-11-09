# ======================================================
# 🤖 AI STOCK PREDICTOR (FinBERT + XGBoost + Streamlit)
# ======================================================
# Author: Parth Khandelwal
# Includes: Multi-Timeframe Candlestick Charts, FinBERT Sentiment, XGBoost Prediction
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
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")
st.title("🤖 AI Stock Predictor (FinBERT + XGBoost + Streamlit)")
st.markdown("A professional dashboard for **real-time stock analysis** with **multi-timeframe charts**, **FinBERT sentiment**, and **AI prediction**.")

# ---------------------------------------
# LOAD FINBERT MODEL
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

symbol = st.sidebar.text_input("Stock Symbol (e.g. RELIANCE.NS, TCS.NS, INFY.NS)", "RELIANCE.NS")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["1m", "5m", "15m", "1h", "4h", "1d", "1wk"],
    index=6,
    format_func=lambda x: f"{x.upper()} (interval)"
)

start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

chart_days = st.sidebar.selectbox(
    "Display Range (Days)",
    options=[90, 180, 365, 730],
    format_func=lambda x: f"Last {x} Days"
)

# ---------------------------------------
# FETCH DATA FUNCTION (fixed for all intervals)
# ---------------------------------------
@st.cache_data(show_spinner=True)
def fetch_data(symbol, start, end, interval):
    try:
        now = datetime.datetime.now()

        # Auto-adjust based on interval
        if interval == "1m":
            start = now - datetime.timedelta(days=7)
        elif interval in ["5m", "15m", "30m", "1h", "4h"]:
            start = now - datetime.timedelta(days=60)
        elif interval == "1d":
            start = now - datetime.timedelta(days=365 * 3)
        elif interval == "1wk":
            start = now - datetime.timedelta(days=365 * 10)

        data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.reset_index()
        if "Datetime" in data.columns:
            data.rename(columns={"Datetime": "Date"}, inplace=True)
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        return data
    except Exception as e:
        st.error(f"⚠️ Data fetch error: {e}")
        return pd.DataFrame()

# ---------------------------------------
# MAIN DASHBOARD
# ---------------------------------------
try:
    df = fetch_data(symbol, start_date, end_date, timeframe)

    if df.empty:
        st.warning(f"⚠️ No data found for {symbol} at timeframe {timeframe.upper()}. Try a higher timeframe.")
    else:
        st.subheader(f"📊 {symbol} ({timeframe.upper()}) Price Chart")

        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()

        df_display = df.tail(chart_days) if len(df) > chart_days else df.copy()

        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df_display["Date"],
            open=df_display["Open"],
            high=df_display["High"],
            low=df_display["Low"],
            close=df_display["Close"],
            name=f"{timeframe.upper()} Candles",
            increasing_line_color="limegreen",
            decreasing_line_color="red",
            increasing_fillcolor="rgba(0,255,0,0.4)",
            decreasing_fillcolor="rgba(255,0,0,0.4)"
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=df_display["Date"], y=df_display["MA5"], mode="lines",
            name="MA5", line=dict(color="cyan", width=1.3)
        ))
        fig.add_trace(go.Scatter(
            x=df_display["Date"], y=df_display["MA20"], mode="lines",
            name="MA20", line=dict(color="orange", width=1.3)
        ))

        # Volume
        fig.add_trace(go.Bar(
            x=df_display["Date"], y=df_display["Volume"]/1e6,
            name="Volume (M)", marker_color="gray", opacity=0.3, yaxis="y2"
        ))

        fig.update_layout(
            title=f"{symbol} — {timeframe.upper()} Trend",
            xaxis_title="Date / Time",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=650,
            xaxis_rangeslider_visible=False,
            yaxis2=dict(overlaying="y", side="right", title="Volume (M)", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # =======================================================
        # 🧠 FINBERT SENTIMENT ANALYSIS
        # =======================================================
        st.subheader("🧠 News Sentiment Analysis")
        news_text = st.text_area("Enter stock-related news or tweet:")

        if st.button("Analyze Sentiment"):
            if news_text.strip():
                sentiment, scores = get_finbert_sentiment(news_text)
                st.success(f"Predicted Sentiment: **{sentiment}**")

                sentiment_df = pd.DataFrame({
                    "Sentiment": ["Negative", "Neutral", "Positive"],
                    "Score": scores
                })
                fig_sent = px.bar(sentiment_df, x="Sentiment", y="Score", color="Sentiment",
                                  title="FinBERT Sentiment Scores", text="Score")
                fig_sent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig_sent.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")

        # =======================================================
        # 🤖 XGBOOST PRICE PREDICTION
        # =======================================================
        st.subheader("📈 Predict Next-Day Movement (Daily Model)")

        daily_data = fetch_data(symbol, start_date, end_date, "1d")
        daily_data["Return"] = daily_data["Close"].pct_change()
        daily_data.dropna(inplace=True)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(daily_data[["Open", "High", "Low", "Close", "Volume"]])
        y = np.where(daily_data["Return"].shift(-1) > 0, 1, 0)

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
            st.success("📈 Prediction: The stock might **rise tomorrow**.")
        else:
            st.error("📉 Prediction: The stock might **fall tomorrow**.")

except Exception as e:
    st.error(f"An error occurred: {e}")
