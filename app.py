# ======================================================
# 🤖 AI STOCK PREDICTOR (FinBERT + XGBoost + Streamlit)
# ======================================================
# Built by Parth Khandelwal
# Features: Stock Price Prediction, Sentiment Analysis, Candlestick Chart
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
st.markdown("An intelligent app that predicts **next-day stock movement** and analyzes **financial news sentiment** using AI.")

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
st.sidebar.header("📊 Stock Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS, TCS.NS)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# ---------------------------------------
# FETCH STOCK DATA
# ---------------------------------------
@st.cache_data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    return data

try:
    df = fetch_data(symbol, start_date, end_date)

    if df.empty:
        st.error("⚠️ No stock data found. Try a different symbol (e.g. RELIANCE.NS).")
    else:
        st.subheader(f"📈 Stock Data for {symbol}")
        st.dataframe(df.tail())

        # =======================================================
        # 📊 FIXED CANDLESTICK CHART (Handles NaN + Datatypes)
        # =======================================================
        st.subheader("📊 Price Chart")

        # Ensure numeric data
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing data
        df = df.dropna(subset=["Open", "High", "Low", "Close"])

        if not df.empty:
            df["MA5"] = df["Close"].rolling(5).mean()
            df["MA20"] = df["Close"].rolling(20).mean()

            # Create the candlestick chart
            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candlestick",
                increasing_line_color="green",
                decreasing_line_color="red"
            ))

            # Moving averages
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["MA5"], mode="lines",
                name="MA5", line=dict(color="blue", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["MA20"], mode="lines",
                name="MA20", line=dict(color="orange", width=1)
            ))

            # Add volume bars below
            fig.add_trace(go.Bar(
                x=df["Date"], y=df["Volume"] / 1e6,
                name="Volume (in millions)", marker_color="gray", opacity=0.3, yaxis="y2"
            ))

            fig.update_layout(
                title=f"{symbol} Price Trend",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    title="Volume (M)",
                    showgrid=False
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No valid stock data found to plot candlesticks.")

        # =======================================================
        # 🧠 SENTIMENT ANALYSIS
        # =======================================================
        st.subheader("🧠 News Sentiment Analysis (FinBERT)")
        news_text = st.text_area("Enter a recent stock-related news headline or tweet:")

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
                st.warning("Please enter some news text for sentiment analysis.")

        # =======================================================
        # 🤖 PRICE PREDICTION MODEL
        # =======================================================
        st.subheader("📈 Predict Next-Day Stock Movement")

        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(df[["Open", "High", "Low", "Close", "Volume"]])
        y = np.where(df["Return"].shift(-1) > 0, 1, 0)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_xgb = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model_xgb.fit(X_train, y_train)

        acc = model_xgb.score(X_test, y_test)
        st.metric("📊 Model Accuracy", f"{acc * 100:.2f}%")

        latest_data = X[-1].reshape(1, -1)
        prediction = model_xgb.predict(latest_data)[0]

        if prediction == 1:
            st.success("📈 Prediction: The stock might **rise tomorrow**.")
        else:
            st.error("📉 Prediction: The stock might **fall tomorrow**.")

except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
