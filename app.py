import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from transformers import pipeline
import requests
import time

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

# ------------------- SIDEBAR -------------------
st.sidebar.header("⚙️ Stock Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, AAPL):", "RELIANCE.NS")

today = datetime.today()
start_default = today - timedelta(days=180)
start_date = st.sidebar.date_input("Start Date", start_default)
end_date = st.sidebar.date_input("End Date", today)

timeframes = {
    "1 Minute": "1",
    "5 Minutes": "5",
    "15 Minutes": "15",
    "1 Hour": "60",
    "4 Hours": "240",
    "1 Day": "D",
    "1 Week": "W"
}
tf_label = st.sidebar.selectbox("Select Timeframe", list(timeframes.keys()), index=6)
interval = timeframes[tf_label]

# Auto refresh every 10 minutes
st.sidebar.markdown("⏱️ Auto-refresh every 10 minutes is enabled")
st_autorefresh = st.experimental_rerun if "autorefresh" in st.session_state else None

st.sidebar.markdown("---")
st.sidebar.caption("💡 Built by Parth Khandelwal")

# ------------------- SYMBOL CONVERSION -------------------
def convert_symbol_for_tradingview(symbol):
    """Convert Yahoo-style symbols to TradingView format."""
    if symbol.endswith(".NS"):
        return f"NSE:{symbol.split('.')[0]}"
    elif symbol.endswith(".BO"):
        return f"BSE:{symbol.split('.')[0]}"
    elif symbol.endswith(".L"):
        return f"LSE:{symbol.split('.')[0]}"
    elif symbol.endswith(".T"):
        return f"TSE:{symbol.split('.')[0]}"
    else:
        return f"NASDAQ:{symbol.upper()}"

symbol_tradingview = convert_symbol_for_tradingview(symbol)

# ------------------- TITLE -------------------
st.title("📈 AI Stock Price Dashboard (Full TradingView + Sentiment AI)")
st.markdown(f"📊 **Live data for {symbol} ({tf_label})**")

# ------------------- FULL TRADINGVIEW CHART -------------------
st.subheader("📊 Interactive TradingView Chart")

st.markdown(
    f"""
    <iframe 
        src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_chart&symbol={symbol_tradingview}&interval={interval}&theme=dark&style=1&locale=en&toolbarbg=f1f3f6&enable_publishing=false&hide_top_toolbar=false&hide_legend=false&save_image=false&calendar=1&studies=[]&width=100%25&height=750"
        width="100%" height="750" frameborder="0" allowtransparency="true" scrolling="no">
    </iframe>
    """,
    unsafe_allow_html=True
)

# ------------------- SENTIMENT ANALYSIS -------------------
st.markdown("---")
st.subheader("🧠 AI-Powered Sentiment Analysis")

sentiment_analyzer = pipeline("sentiment-analysis")

# 1️⃣ Manual Input
st.markdown("### ✍️ Analyze Custom News or Tweet")
news_input = st.text_area("Enter a stock-related headline:", placeholder="Example: Reliance Industries reports record profits...")

if st.button("🔍 Analyze This"):
    if news_input.strip():
        result = sentiment_analyzer(news_input)[0]
        label, score = result['label'], result['score']
        if label == "POSITIVE":
            st.success(f"📈 Bullish Sentiment — Confidence: {score*100:.2f}%")
        elif label == "NEGATIVE":
            st.error(f"📉 Bearish Sentiment — Confidence: {score*100:.2f}%")
        else:
            st.info(f"⚖️ Neutral Sentiment — Confidence: {score*100:.2f}%")
    else:
        st.warning("Please enter some text before analyzing.")

# 2️⃣ Auto News Sentiment
st.markdown("### 🗞️ Auto-Fetch Latest Headlines")
st.caption("Pulling real-time financial news and analyzing sentiment...")

try:
    # Replace with your own NewsAPI key
    news_api_url = f"https://newsapi.org/v2/everything?q={symbol.split('.')[0]}&language=en&sortBy=publishedAt&pageSize=5&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(news_api_url)
    news_data = response.json()

    if "articles" in news_data and len(news_data["articles"]) > 0:
        articles = news_data["articles"][:5]
        sentiments = []
        st.write("#### 📰 Top Recent Headlines:")
        for article in articles:
            title = article["title"]
            result = sentiment_analyzer(title)[0]
            label = result['label']
            score = result['score']
            sentiments.append(score if label == "POSITIVE" else -score)

            if label == "POSITIVE":
                st.success(f"📈 {title}")
            elif label == "NEGATIVE":
                st.error(f"📉 {title}")
            else:
                st.info(f"⚖️ {title}")

        avg_sentiment = sum(sentiments) / len(sentiments)
        st.markdown("---")
        if avg_sentiment > 0.2:
            st.success(f"🟢 Overall Market Mood: **Bullish** ({avg_sentiment*100:.1f}%)")
        elif avg_sentiment < -0.2:
            st.error(f"🔴 Overall Market Mood: **Bearish** ({avg_sentiment*100:.1f}%)")
        else:
            st.info(f"⚪ Neutral Market Mood ({avg_sentiment*100:.1f}%)")

    else:
        st.warning("⚠️ No recent headlines found for this stock.")

except Exception as e:
    st.warning("⚠️ Could not fetch live news automatically. Please enter your own headline instead.")
    st.caption(f"Error: {e}")

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("🚀 Created by Parth Khandelwal • Powered by Streamlit, HuggingFace Transformers & TradingView")

# ------------------- AUTO REFRESH -------------------
# Refresh every 10 minutes automatically
st.experimental_set_query_params(refresh=int(time.time()))
st_autorefresh_interval = 600000  # 10 minutes in ms
st.markdown(
    f"""
    <script>
    setTimeout(function(){{
        window.location.reload(1);
    }}, {st_autorefresh_interval});
    </script>
    """,
    unsafe_allow_html=True
)
