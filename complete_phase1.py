import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import logging
import plotly.graph_objects as go
import socket
import warnings
import requests
import pytz

# Suppress deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Additional libraries for live data and options
try:
    from yahoo_fin import options as yf_options
except ImportError:
    yf_options = None
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
try:
    import finnhub
except ImportError:
    finnhub = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- Custom CSS for a Stock Market Vibe ----------
custom_css = """
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #2E3B4E; color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, h5, h6 { color: #2E3B4E; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stTabs [role="tab"] { background: #4CAF50; color: white; }
    .stTabs [role="tab"]:hover { background: #45a049; }
    .metric-label { font-weight: bold; color: #2E3B4E; }
    .metric-value { color: #4CAF50; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- Helper Functions ----------

def is_market_open(ticker_sym):
    """Returns True if the market for the given ticker is currently open."""
    now_utc = datetime.datetime.now(pytz.utc)
    if ticker_sym.endswith(".NS") or ticker_sym.endswith(".BO"):
        # Indian markets: 9:15 AM to 3:30 PM IST, Monday-Friday
        ist = now_utc.astimezone(pytz.timezone("Asia/Kolkata"))
        open_time = ist.replace(hour=9, minute=15, second=0, microsecond=0)
        close_time = ist.replace(hour=15, minute=30, second=0, microsecond=0)
        return ist.weekday() < 5 and open_time <= ist <= close_time
    else:
        # US markets: 9:30 AM to 4:00 PM ET, Monday-Friday
        et = now_utc.astimezone(pytz.timezone("US/Eastern"))
        open_time = et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = et.replace(hour=16, minute=0, second=0, microsecond=0)
        return et.weekday() < 5 and open_time <= et <= close_time

def get_nse_live_quote(symbol):
    """
    Fetch live NSE/BSE quote data by calling NSE's JSON endpoint.
    'symbol' should be provided without suffix (e.g., 'RELIANCE').
    Returns a JSON dictionary.
    """
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
         "Accept-Language": "en-US,en;q=0.9",
         "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    session = requests.Session()
    session.headers.update(headers)
    session.get("https://www.nseindia.com", timeout=5)
    response = session.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# ---------- Sidebar: Company, Date, and Advanced Options ----------
st.sidebar.title("Company & Live Data Settings")
st.sidebar.markdown("Enter a company ticker and date range. Expand Advanced Options for live data source settings.")

# Predefined companies – for Indian stocks, use proper suffix (.NS for NSE, .BO for BSE)
companies = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corporation (MSFT)": "MSFT",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Tesla Inc. (TSLA)": "TSLA",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.BO)": "TCS.BO",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
}

input_method = st.sidebar.radio("Choose Input Method", ("Select from list", "Enter ticker manually"))
if input_method == "Select from list":
    company_name = st.sidebar.selectbox("Select Company", list(companies.keys()))
    ticker_symbol = companies[company_name]
else:
    ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")

# Instead of date pickers, type the dates in YYYY-MM-DD format
start_date_str = st.sidebar.text_input("Start Date (YYYY-MM-DD)", value="2020-01-01")
end_date_str = st.sidebar.text_input("End Date (YYYY-MM-DD)", value=datetime.date.today().strftime("%Y-%m-%d"))

try:
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
except Exception as e:
    st.error(f"Error parsing dates: {e}")
    st.stop()

if start_date == end_date:
    st.sidebar.warning("Same start and end date entered. Switching to intraday mode.")
    intraday_interval = st.sidebar.selectbox("Select Intraday Interval", 
                                              options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"])
else:
    intraday_interval = None

with st.sidebar.expander("Advanced Options"):
    live_source = st.radio("For US live data, choose source:", 
                           ("Ask me later", "Use Finnhub (real-time)", "Use yfinance (delayed)"))
    finnhub_api_key = ""
    if live_source == "Use Finnhub (real-time)":
        finnhub_api_key = st.text_input("Enter Finnhub API Key", value="", type="password")

# ---------- Data Fetching from yfinance ----------
@st.cache_data(show_spinner=True, ttl=3600)
def fetch_all_data(ticker_sym, start, end, interval=None):
    """Fetch available data from yfinance for the given ticker."""
    ticker = yf.Ticker(ticker_sym)
    try:
        info = ticker.info
    except Exception as e:
        logging.error("Failed to fetch ticker info: %s", e)
        info = {}
    try:
        if interval:
            end_dt = start + datetime.timedelta(days=1)
            history = ticker.history(start=start, end=end_dt, interval=interval)
        else:
            history = ticker.history(start=start, end=end)
    except Exception as e:
        logging.error("Failed to fetch historical data: %s", e)
        history = pd.DataFrame()
    try:
        dividends = ticker.dividends
    except Exception as e:
        logging.error("Failed to fetch dividends: %s", e)
        dividends = pd.Series(dtype='float64')
    try:
        splits = ticker.splits
    except Exception as e:
        logging.error("Failed to fetch splits: %s", e)
        splits = pd.Series(dtype='float64')
    try:
        income_statement = ticker.financials
    except Exception as e:
        logging.warning("Income statement not available: %s", e)
        income_statement = pd.DataFrame()
    try:
        balance_sheet = ticker.balance_sheet
    except Exception as e:
        logging.warning("Balance sheet not available: %s", e)
        balance_sheet = pd.DataFrame()
    try:
        cash_flow = ticker.cashflow
    except Exception as e:
        logging.warning("Cash flow not available: %s", e)
        cash_flow = pd.DataFrame()
    try:
        earnings = ticker.earnings
        if earnings is None:
            earnings = pd.DataFrame()
    except Exception as e:
        logging.warning("Earnings not available: %s", e)
        earnings = pd.DataFrame()
    try:
        quarterly_earnings = ticker.quarterly_earnings
        if quarterly_earnings is None:
            quarterly_earnings = pd.DataFrame()
    except Exception as e:
        logging.warning("Quarterly earnings not available: %s", e)
        quarterly_earnings = pd.DataFrame()
    try:
        sustainability = ticker.sustainability
        if sustainability is None:
            sustainability = pd.DataFrame()
        else:
            sustainability = sustainability.astype(str)
    except Exception as e:
        logging.warning("Sustainability not available: %s", e)
        sustainability = pd.DataFrame()
    try:
        major_holders = ticker.major_holders
        if major_holders is None:
            major_holders = pd.DataFrame()
    except Exception as e:
        logging.warning("Major holders not available: %s", e)
        major_holders = pd.DataFrame()
    try:
        institutional_holders = ticker.institutional_holders
        if institutional_holders is None:
            institutional_holders = pd.DataFrame()
    except Exception as e:
        logging.warning("Institutional holders not available: %s", e)
        institutional_holders = pd.DataFrame()
    try:
        calendar = ticker.calendar
        if isinstance(calendar, dict):
            calendar = pd.DataFrame.from_dict(calendar, orient='index', columns=["Value"])
        elif calendar is None:
            calendar = pd.DataFrame()
    except Exception as e:
        logging.warning("Calendar not available: %s", e)
        calendar = pd.DataFrame()
    try:
        options_dates = ticker.options
        if options_dates is None:
            options_dates = []
    except Exception as e:
        logging.warning("Options dates not available: %s", e)
        options_dates = []
    return {
        "info": info,
        "history": history,
        "dividends": dividends,
        "splits": splits,
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "earnings": earnings,
        "quarterly_earnings": quarterly_earnings,
        "sustainability": sustainability,
        "major_holders": major_holders,
        "institutional_holders": institutional_holders,
        "calendar": calendar,
        "options_dates": options_dates,
    }

# Check network connectivity
try:
    socket.gethostbyname("query1.finance.yahoo.com")
except socket.gaierror:
    st.error("Network error: Unable to resolve 'query1.finance.yahoo.com'. Check your internet/DNS settings.")
    st.stop()

with st.spinner("Fetching company data..."):
    data = fetch_all_data(ticker_symbol, start_date, end_date, intraday_interval)

# ---------- Main Dashboard Layout ----------
st.title("Company Information Dashboard")
st.markdown(f"### Detailed Data for **{ticker_symbol}**")
st.markdown("---")

# Define tabs (removing earnings tab as requested)
tabs = st.tabs([
    "Overview", "Historical Data", "Dividends & Splits", "Financials",
    "Options", "Holders", "Sustainability", "Calendar", "Live Data"
])

# ---------- Overview Tab ----------
with tabs[0]:
    st.header("Fundamental Information")
    st.markdown("This section displays core details about the company (e.g., name, sector, market cap).")
    info = data["info"]
    if info:
        keys_to_show = [
            "longName", "sector", "industry", "marketCap", "previousClose",
            "open", "volume", "regularMarketPrice", "trailingPE", "dividendYield"
        ]
        for key in keys_to_show:
            value = info.get(key, "N/A")
            st.markdown(f"**{key}:** {value}")
        st.markdown("---")
        st.subheader("Full Info Dictionary")
        st.json(info)
    else:
        st.write("No fundamental information available.")

# ---------- Historical Data Tab ----------
with tabs[1]:
    st.header("Historical Price Data")
    st.markdown("This tab displays historical price data. Use the download button below for CSV export.")
    history = data["history"]
    if not history.empty:
        st.dataframe(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history["Close"],
                                 mode='lines', name="Close Price"))
        fig.update_layout(title=f"{ticker_symbol} Close Price History",
                          xaxis_title="Date/Time",
                          yaxis_title="Price ($)",
                          template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Historical Data as CSV",
            data=history.to_csv().encode("utf-8"),
            file_name=f"{ticker_symbol}_historical_data.csv",
            mime="text/csv"
        )
    else:
        st.write("No historical data available.")

# ---------- Dividends & Splits Tab ----------
with tabs[2]:
    st.header("Dividends & Splits")
    st.markdown("Dividend and split histories can explain changes in stock price over time.")
    dividends = data["dividends"]
    splits = data["splits"]
    if not dividends.empty:
        st.subheader("Dividends")
        st.dataframe(dividends)
    else:
        st.write("No dividend data available.")
    if not splits.empty:
        st.subheader("Splits")
        st.dataframe(splits)
    else:
        st.write("No split data available.")

# ---------- Financials Tab ----------
with tabs[3]:
    st.header("Financial Statements")
    st.markdown("Review the company's financial statements (Income Statement, Balance Sheet, Cash Flow) to assess financial health.")
    income_statement = data["income_statement"]
    balance_sheet = data["balance_sheet"]
    cash_flow = data["cash_flow"]
    if not income_statement.empty:
        st.subheader("Income Statement")
        st.dataframe(income_statement)
    else:
        st.write("Income statement not available.")
    if not balance_sheet.empty:
        st.subheader("Balance Sheet")
        st.dataframe(balance_sheet)
    else:
        st.write("Balance sheet not available.")
    if not cash_flow.empty:
        st.subheader("Cash Flow Statement")
        st.dataframe(cash_flow)
    else:
        st.write("Cash flow statement not available.")

# ---------- Options Tab ----------
with tabs[4]:
    st.header("Options Data")
    st.markdown("Options data can provide insights into market expectations. This tab uses the yahoo_fin module for options data.")
    if yf_options:
        try:
            opt_chain = yf_options.get_options_chain(ticker_symbol)
            calls = opt_chain.get("calls", pd.DataFrame())
            puts = opt_chain.get("puts", pd.DataFrame())
            # Convert all columns to string to avoid Arrow serialization errors
            if not calls.empty:
                calls = calls.astype(str)
            if not puts.empty:
                puts = puts.astype(str)
            st.markdown("**Calls:**")
            if not calls.empty:
                st.dataframe(calls)
            else:
                st.write("No call options data available.")
            st.markdown("**Puts:**")
            if not puts.empty:
                st.dataframe(puts)
            else:
                st.write("No put options data available.")
        except Exception as e:
            st.error(f"Error fetching options chain using yahoo_fin: {e}")
    else:
        st.error("yahoo_fin.options module not available. Install via 'pip install yahoo_fin'.")

# ---------- Holders Tab ----------
with tabs[5]:
    st.header("Holders Information")
    st.markdown("Major and institutional holders indicate insider and institutional confidence in the stock.")
    major_holders = data["major_holders"]
    institutional_holders = data["institutional_holders"]
    if not major_holders.empty:
        st.subheader("Major Holders")
        st.dataframe(major_holders)
    else:
        st.write("Major holders data not available.")
    if not institutional_holders.empty:
        st.subheader("Institutional Holders")
        st.dataframe(institutional_holders)
    else:
        st.write("Institutional holders data not available.")

# ---------- Sustainability Tab ----------
with tabs[6]:
    st.header("Sustainability / ESG Metrics")
    st.markdown("ESG metrics help assess a company's ethical and operational performance. Higher scores usually imply lower risk. Additional metrics (e.g., controversy ratings) may also be available.")
    sustainability = data["sustainability"]
    if not sustainability.empty:
        st.dataframe(sustainability)
    else:
        st.write("Sustainability data not available.")

# ---------- Calendar Tab ----------
with tabs[7]:
    st.header("Company Calendar")
    st.markdown("The calendar shows upcoming events (e.g., earnings announcements) that may affect the stock price.")
    calendar = data["calendar"]
    if not calendar.empty:
        st.dataframe(calendar)
    else:
        st.write("Calendar data not available.")

# ---------- Live Data Tab ----------
with tabs[8]:
    st.header("Live Data")
    # Check if the market is open
    market_open = is_market_open(ticker_symbol)
    if not market_open:
        st.warning("Market is currently closed. Data below may be delayed.")
    st.markdown("This tab displays live market data. It auto-refreshes every 10 seconds.")
    if st_autorefresh:
        st_autorefresh(interval=10000, key="liveData")
    else:
        st.info("Install streamlit-autorefresh to enable auto-refresh.")
    
    # For Indian stocks, try to fetch live data using our custom NSE scraper.
    if ticker_symbol.endswith(".NS") or ticker_symbol.endswith(".BO"):
        try:
            symbol = ticker_symbol.split('.')[0]
            quote = get_nse_live_quote(symbol)
            price_info = quote.get("priceInfo", {})
            live_price = price_info.get("lastPrice", "N/A")
            pChange = price_info.get("pChange", "N/A")
            st.metric("Live Price", f"₹{live_price}", delta=f"{pChange}%")
            # Show a live intraday chart using yfinance as backup (delayed)
            live_history = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
            if not live_history.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=live_history.index, y=live_history["Close"],
                                         mode="lines", name="Live Price"))
                fig.update_layout(title=f"{ticker_symbol} Live Intraday Price",
                                  xaxis_title="Time",
                                  yaxis_title="Price (₹)",
                                  template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No live intraday chart available.")
        except Exception as e:
            st.error(f"Error fetching live data from NSE/BSE: {e}")
            try:
                live_history = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
                if not live_history.empty:
                    last_row = live_history.iloc[-1]
                    live_price = last_row["Close"]
                    st.metric("Live Price (Delayed)", f"₹{live_price}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=live_history.index, y=live_history["Close"],
                                             mode="lines", name="Live Price"))
                    fig.update_layout(title=f"{ticker_symbol} Live Intraday Price (Delayed)",
                                      xaxis_title="Time",
                                      yaxis_title="Price (₹)",
                                      template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No live data available.")
            except Exception as e:
                st.error(f"Error fetching intraday data from yfinance: {e}")
    else:
        # For US stocks, check user's selection from Advanced Options.
        if live_source == "Use Finnhub (real-time)" and finnhub and finnhub_api_key:
            try:
                finnhub_client = finnhub.Client(api_key=finnhub_api_key)
                quote = finnhub_client.quote(ticker_symbol)
                live_price = quote.get("c", "N/A")
                change_percent = quote.get("dp", "N/A")
                st.metric("Live Price", f"${live_price}", delta=f"{change_percent}%")
                # Also display a live intraday chart from yfinance as backup (delayed)
                live_history = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
                if not live_history.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=live_history.index, y=live_history["Close"],
                                             mode="lines", name="Live Price"))
                    fig.update_layout(title=f"{ticker_symbol} Live Intraday Price (Delayed)",
                                      xaxis_title="Time",
                                      yaxis_title="Price ($)",
                                      template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No live intraday chart available.")
            except Exception as e:
                st.error(f"Error fetching live data from Finnhub: {e}")
        elif live_source == "Use yfinance (delayed)":
            try:
                live_history = yf.Ticker(ticker_symbol).history(period="1d", interval="1m")
                if not live_history.empty:
                    last_row = live_history.iloc[-1]
                    live_price = last_row["Close"]
                    st.metric("Live Price (Delayed)", f"${live_price}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=live_history.index, y=live_history["Close"],
                                             mode="lines", name="Live Price"))
                    fig.update_layout(title=f"{ticker_symbol} Live Intraday Price (Delayed)",
                                      xaxis_title="Time",
                                      yaxis_title="Price ($)",
                                      template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No live data available.")
            except Exception as e:
                st.error(f"Error fetching live data from yfinance: {e}")
        else:
            st.info("Please select a live data source for US stocks in the Advanced Options.")


