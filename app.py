import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline
import requests

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# ------------------- SIDEBAR -------------------
st.sidebar.header("⚙️ Stock Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")

today = datetime.today()
start_default = today - timedelta(days=180)
start_date = st.sidebar.date_input("Start Date", start_default)
end_date = st.sidebar.date_input("End Date", today)

timeframes = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "1 Hour": "60m",
    "4 Hours": "240m",
    "1 Day": "1d",
    "1 Week": "1wk"
}
tf_label = st.sidebar.selectbox("Select Timeframe", list(timeframes.keys()), index=6)
interval = timeframes[tf_label]

st.sidebar.markdown("---")
st.sidebar.caption("💡 Built by Parth Khandelwal")

# ------------------- TITLE -------------------
st.title("📈 AI Stock Price Dashboard (TradingView + Sentiment AI)")
st.markdown(f"📊 **Showing data for {symbol} ({interval})**")

# ------------------- FETCH DATA -------------------
try:
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

    if df.empty:
        st.warning("⚠️ No data found for this symbol or timeframe. Try another combination.")
    else:
        df.reset_index(inplace=True)
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        col1, col2 = st.columns([2, 1])

        # ------------------- PLOTLY CANDLESTICK -------------------
        with col1:
            st.subheader(f"{symbol} — {tf_label} Candlestick Chart")

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
                open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name=f"{tf_label} Candles"
            ))

            fig.add_trace(go.Scatter(
                x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
                y=df["MA5"], mode="lines", line=dict(color="cyan", width=1.2), name="MA5"
            ))

            fig.add_trace(go.Scatter(
                x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
                y=df["MA20"], mode="lines", line=dict(color="orange", width=1.2), name="MA20"
            ))

            fig.update_layout(
                template="plotly_dark",
                height=600,
                xaxis_title="Date / Time",
                yaxis_title="Price (₹)",
                yaxis2=dict(title="Volume (M)", overlaying="y", side="right", showgrid=False),
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        # ------------------- TRADINGVIEW EMBED -------------------
        with col2:
            st.subheader("📊 Live TradingView Chart")
            interval_embed = (
                interval.replace("m", "") if "m" in interval else
                interval.replace("wk", "1W") if "wk" in interval else
                interval
            )

            st.markdown(
                f"""
                <iframe 
                    src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_abc12&symbol={symbol}&interval={interval_embed}&theme=dark&style=1&locale=en&toolbarbg=f1f3f6&hide_top_toolbar=0&hide_legend=1&save_image=0&calendar=1&studies=[]"
                    width="100%" height="600" frameborder="0" allowtransparency="true" scrolling="no">
                </iframe>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"❌ Error loading data: {e}")

# ------------------- SENTIMENT ANALYSIS -------------------
st.markdown("---")
st.subheader("🧠 AI-Powered Sentiment Analysis")

news_input = st.text_area("📰 Enter latest stock-related news or tweet:", placeholder="Example: Reliance Industries reports record quarterly profits...")

if st.button("🔍 Analyze Sentiment"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text or news headline first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment_analyzer = pipeline("sentiment-analysis")
                result = sentiment_analyzer(news_input)[0]

                sentiment = result['label']
                confidence = result['score']

                if sentiment == "POSITIVE":
                    st.success(f"📈 **Bullish Sentiment** — Confidence: {confidence*100:.2f}%")
                elif sentiment == "NEGATIVE":
                    st.error(f"📉 **Bearish Sentiment** — Confidence: {confidence*100:.2f}%")
                else:
                    st.info(f"⚖️ **Neutral Sentiment** — Confidence: {confidence*100:.2f}%")

            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {e}")

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("🚀 Created by Parth Khandelwal • Powered by Streamlit, Yahoo Finance, HuggingFace Transformers & TradingView")
