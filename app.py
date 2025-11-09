# ======================================================
# 🤖 AI STOCK PREDICTOR (FinBERT + XGBoost + Streamlit)
# ======================================================
# Features:
# ✅ Multi-timeframe charts (1m–1wk)
# ✅ FinBERT sentiment analysis
# ✅ XGBoost next-day prediction
# ✅ Auto-refresh every 1 min for live charts
# ✅ Instant manual refresh for date/timeframe changes
# ✅ Dark/Light theme toggle
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
import time

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="wide")

st.title("🤖 AI Stock Predictor (Live + FinBERT + XGBoost)")
st.markdown("A **live, interactive dashboard** combining technical charts, FinBERT sentiment & XGBoost AI forecasting.")

# ---------------------------------------
# THEME SWITCH
# ---------------------------------------
theme = st.sidebar.radio("🌓 Select Theme", ["Dark", "Light"], horizontal=True)
plot_theme = "plotly_dark" if theme == "Dark" else "plotly_white"

# ---------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------
st.sidebar.header("⚙️ Settings")

symbol = st.sidebar.text_input("Stock Symbol (e.g. RELIANCE.NS, TCS.NS, INFY.NS)", "RELIANCE.NS")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["1m", "5m", "15m", "1h", "4h", "1d", "1wk"],
    index=6
)

start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

chart_days = st.sidebar.selectbox(
    "Display Range (Days)",
    options=[30, 90, 180, 365],
    format_func=lambda x: f"Last {x} Days"
)

auto_refresh = st.sidebar.checkbox("🔁 Auto-Refresh every 1 minute", value=(timeframe in ["1m", "5m"]))

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
# FETCH DATA FUNCTION
# ---------------------------------------
@st.cache_data(ttl=60, show_spinner=True)
def fetch_data(symbol, start, end, interval):
    now = datetime.datetime.now()
    if interval == "1m":
        start = now - datetime.timedelta(days=7)
    elif interval in ["5m", "15m", "1h", "4h"]:
        start = now - datetime.timedelta(days=60)
    elif interval == "1d":
        start = now - datetime.timedelta(days=365 * 3)
    elif interval == "1wk":
        start = now - datetime.timedelta(days=365 * 10)

    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------------------------------------
# DRAW CHART FUNCTION
# ---------------------------------------
def draw_chart(df, symbol, timeframe, plot_theme, chart_days):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df_display = df.tail(chart_days) if len(df) > chart_days else df.copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_display["Date"], open=df_display["Open"], high=df_display["High"],
        low=df_display["Low"], close=df_display["Close"],
        name=f"{timeframe.upper()} Candles",
        increasing_line_color="limegreen", decreasing_line_color="red"
    ))

    fig.add_trace(go.Scatter(
        x=df_display["Date"], y=df_display["MA5"], mode="lines",
        name="MA5", line=dict(color="cyan", width=1.3)
    ))
    fig.add_trace(go.Scatter(
        x=df_display["Date"], y=df_display["MA20"], mode="lines",
        name="MA20", line=dict(color="orange", width=1.3)
    ))

    fig.add_trace(go.Bar(
        x=df_display["Date"], y=df_display["Volume"]/1e6,
        name="Volume (M)", marker_color="gray", opacity=0.3, yaxis="y2"
    ))

    fig.update_layout(
        template=plot_theme,
        height=650,
        title=f"{symbol} ({timeframe.upper()}) Price Trend",
        xaxis_title="Date / Time",
        yaxis_title="Price (INR)",
        yaxis2=dict(overlaying="y", side="right", title="Volume (M)", showgrid=False),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------------------------------------
# MAIN DASHBOARD LOGIC
# ---------------------------------------
try:
    df = fetch_data(symbol, start_date, end_date, timeframe)

    if df.empty:
        st.warning(f"⚠️ No data found for {symbol} ({timeframe.upper()}). Try a higher timeframe.")
    else:
        fig = draw_chart(df, symbol, timeframe, plot_theme, chart_days)
        st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        st.toast("🔄 Auto-refresh enabled (updates every 60 seconds)")
        time.sleep(60)
        st.experimental_rerun()

except Exception as e:
    st.error(f"An error occurred: {e}")

# ---------------------------------------
# 🧠 FINBERT SENTIMENT
# ---------------------------------------
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
                          title="FinBERT Sentiment Scores", text="Score", template=plot_theme)
        fig_sent.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.warning("Please enter text to analyze.")

# ---------------------------------------
# 🤖 XGBOOST PREDICTION
# ---------------------------------------
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
