import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="In-Depth Company Analysis", layout="wide")
st.title("In-Depth Company Analysis")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 1, 1))

# ---------------- Data Fetching ----------------
@st.cache_data(show_spinner=True)
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    return data

data = fetch_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data found for the selected ticker and date range.")
    st.stop()

st.subheader(f"Historical Data for {ticker.upper()}")
st.dataframe(data.head())

# ---------------- Technical Indicator Calculations ----------------
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

data['STD20'] = data['Close'].rolling(window=20).std()
data['Upper Band'] = data['MA20'] + 2 * data['STD20']
data['Lower Band'] = data['MA20'] - 2 * data['STD20']

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
data['RSI'] = compute_rsi(data['Close'])

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line
data['MACD'], data['Signal'], data['MACD_Hist'] = compute_macd(data['Close'])

data['Daily Return'] = data['Close'].pct_change()
data['Cumulative Return'] = (data['Daily Return'] + 1).cumprod() - 1
data['Price Range'] = data['High'] - data['Low']
corr_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

# ---------------- Prepare DataFrames for Plotting ----------------
df_all = data.reset_index()
# Create a string version of Date for clearer x-axis labels in simple charts
df_all['Date_str'] = df_all['Date'].dt.strftime("%Y-%m-%d")

# Build a dedicated DataFrame for Moving Averages in long format
df_ma_long = pd.DataFrame({
    "Date": df_all["Date_str"].values.ravel(),
    "Close": df_all["Close"].values.ravel(),
    "MA20": df_all["MA20"].values.ravel(),
    "MA50": df_all["MA50"].values.ravel(),
    "MA200": df_all["MA200"].values.ravel()
}).melt(id_vars=["Date"], value_vars=["Close", "MA20", "MA50", "MA200"],
       var_name="Legend", value_name="Price")

# Build a dedicated DataFrame for Bollinger Bands in long format
df_bb_long = pd.DataFrame({
    "Date": df_all["Date_str"].values.ravel(),
    "Close": df_all["Close"].values.ravel(),
    "MA20": df_all["MA20"].values.ravel(),
    "Upper Band": df_all["Upper Band"].values.ravel(),
    "Lower Band": df_all["Lower Band"].values.ravel()
}).melt(id_vars=["Date"], value_vars=["Close", "MA20", "Upper Band", "Lower Band"],
       var_name="Legend", value_name="Price")

# ---------------- Key Metrics Summary ----------------
st.markdown("### Key Metrics Summary")
max_price = float(data['High'].max())
min_price = float(data['Low'].min())
avg_volume = float(data['Volume'].mean())
st.write(f"**Max Price:** ${max_price:.2f}")
st.write(f"**Min Price:** ${min_price:.2f}")
st.write(f"**Average Volume:** {avg_volume:,.0f}")

# ---------------- Narrative Explanation ----------------
def generate_narrative(data):
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    change_pct = ((end_price - start_price) / start_price) * 100
    volatility = float(data['Daily Return'].std() * 100)
    trend = "an upward, bullish trend" if change_pct >= 0 else "a downward, bearish trend"
    return (
        f"From **${start_price:.2f}** to **${end_price:.2f}**, the stock changed by **{abs(change_pct):.2f}%** "
        f"({trend}) with daily volatility of about **{volatility:.2f}%**."
    )
narrative_text = generate_narrative(data)
st.markdown("### Narrative")
st.write(narrative_text)

# ---------------- Interactive Visualizations ----------------

# 1. Trading Volume Chart
st.markdown("#### 1. Trading Volume")
vol_data = {
    "Date": df_all["Date_str"].values.ravel().tolist(),
    "Volume": df_all["Volume"].values.ravel().tolist()
}
fig_volume = px.bar(
    data_frame=pd.DataFrame(vol_data),
    x="Date",
    y="Volume",
    title=f"{ticker.upper()} Trading Volume",
    labels={"Date": "Date", "Volume": "Volume"}
)
st.plotly_chart(fig_volume, use_container_width=True)
st.write("This bar chart shows the daily trading volume, highlighting periods of high market activity.")

# 2. 20-Day Rolling Volatility Chart
st.markdown("#### 2. 20-Day Rolling Volatility")
data['Rolling Volatility'] = data['Daily Return'].rolling(window=20).std() * 100
df_vol = data.reset_index()
df_vol['Date_str'] = df_vol['Date'].dt.strftime("%Y-%m-%d")
fig_vol = px.line(
    data_frame=df_vol,
    x="Date_str",
    y="Rolling Volatility",
    title=f"{ticker.upper()} 20-Day Rolling Volatility",
    labels={"Date_str": "Date", "Rolling Volatility": "Volatility (%)"}
)
st.plotly_chart(fig_vol, use_container_width=True)
st.write("This chart shows the 20-day rolling volatility, giving insight into market variability over time.")

# 3. RSI Chart
st.markdown("#### 3. Relative Strength Index (RSI)")
fig_rsi = px.line(
    data_frame=df_all,
    x="Date_str",
    y="RSI",
    title=f"{ticker.upper()} RSI",
    labels={"Date_str": "Date", "RSI": "RSI Value"}
)
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
st.plotly_chart(fig_rsi, use_container_width=True)
st.write("The RSI chart indicates whether the stock is overbought (above 70) or oversold (below 30).")

# 4. MACD Chart
st.markdown("#### 4. MACD Analysis")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df_all["Date_str"], y=df_all["MACD"], mode="lines", name="MACD"))
fig_macd.add_trace(go.Scatter(x=df_all["Date_str"], y=df_all["Signal"], mode="lines", name="Signal"))
fig_macd.add_trace(go.Bar(x=df_all["Date_str"], y=df_all["MACD_Hist"], name="Histogram"))
fig_macd.update_layout(title=f"{ticker.upper()} MACD", xaxis_title="Date", yaxis_title="MACD")
st.plotly_chart(fig_macd, use_container_width=True)
st.write("The MACD chart compares short- and long-term EMAs to reveal momentum shifts, aiding in potential trading signals.")

# 5. Daily Returns Histogram
st.markdown("#### 5. Daily Returns Distribution")
fig_returns = px.histogram(
    data_frame=df_all,
    x="Daily Return",
    nbins=50,
    title=f"{ticker.upper()} Daily Returns Distribution"
)
fig_returns.update_xaxes(tickangle=90)
st.plotly_chart(fig_returns, use_container_width=True)
st.write("This histogram displays the distribution of daily returns, offering insights into the frequency and magnitude of price changes.")

# 6. Correlation Heatmap
st.markdown("#### 6. Correlation Heatmap")
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix"
)
fig_corr.update_xaxes(tickangle=90)
st.plotly_chart(fig_corr, use_container_width=True)
st.write("The correlation heatmap shows relationships among key variables (Open, High, Low, Close, Volume), revealing interdependencies.")

