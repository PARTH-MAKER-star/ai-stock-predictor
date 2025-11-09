import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Stock Visualizer", layout="wide")
st.title("📊 AI Stock Price Dashboard (TradingView Style)")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Stock Settings")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")

# Date Range
today = datetime.today()
default_start = today - timedelta(days=180)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

# Timeframe options
interval_map = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "1 Hour": "60m",
    "4 Hours": "240m",
    "1 Day": "1d",
    "1 Week": "1wk"
}
timeframe = st.sidebar.selectbox("Select Timeframe", list(interval_map.keys()), index=6)

# --- FETCH DATA ---
st.write(f"📈 **Showing data for {symbol} ({interval_map[timeframe]})**")

try:
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval_map[timeframe])

    if df.empty:
        st.warning("⚠️ No data found for this date range or timeframe.")
    else:
        # Reset index for plotting
        df.reset_index(inplace=True)

        # --- MOVING AVERAGES ---
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        # --- PLOTLY CANDLESTICK CHART ---
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=f"{timeframe} Candles"
        ))

        # Moving averages
        fig.add_trace(go.Scatter(
            x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
            y=df["MA5"], mode="lines", line=dict(color="cyan", width=1),
            name="MA5"
        ))
        fig.add_trace(go.Scatter(
            x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
            y=df["MA20"], mode="lines", line=dict(color="orange", width=1),
            name="MA20"
        ))

        # Volume
        fig.add_trace(go.Bar(
            x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
            y=df["Volume"] / 1_000_000,
            name="Volume (M)", marker_color="gray", opacity=0.3, yaxis="y2"
        ))

        # Layout
        fig.update_layout(
            xaxis_title="Date / Time",
            yaxis_title="Price (₹)",
            yaxis2=dict(
                overlaying="y", side="right", title="Volume (M)", showgrid=False
            ),
            template="plotly_dark",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("🚀 Built with Streamlit + Yahoo Finance + Plotly • Created by Parth Khandelwal")
Updated app.py (final working version)
