import os
import time
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Sentiment Dashboard",
                   layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .main { background-color: #0f0f1a; }
    h1, h2, h3 { color: #00d4ff; }
    .metric-container { background: #1a1a2e; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction & Sentiment Dashboard")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

TICKER_OPTIONS = {
    "🍎 Apple (AAPL)"           : "AAPL",
    "🪟 Microsoft (MSFT)"       : "MSFT",
    "🔍 Alphabet / Google (GOOGL)": "GOOGL",
    "📦 Amazon (AMZN)"          : "AMZN",
    "🤖 NVIDIA (NVDA)"          : "NVDA",
    "🚗 Tesla (TSLA)"           : "TSLA",
    "📱 Meta (META)"            : "META",
    "🎵 Netflix (NFLX)"         : "NFLX",
    "💳 Visa (V)"               : "V",
    "☕ Starbucks (SBUX)"       : "SBUX",
    "🏦 JPMorgan Chase (JPM)"   : "JPM",
    "🏦 Goldman Sachs (GS)"     : "GS",
    "🏦 Bank of America (BAC)"  : "BAC",
    "💊 Johnson & Johnson (JNJ)": "JNJ",
    "💊 Pfizer (PFE)"           : "PFE",
    "⚡ ExxonMobil (XOM)"       : "XOM",
    "⚡ Chevron (CVX)"          : "CVX",
    "✈️ Boeing (BA)"            : "BA",
    "🛒 Walmart (WMT)"          : "WMT",
    "🍔 McDonald's (MCD)"       : "MCD",
    "🧪 ASML (ASML)"            : "ASML",
    "💻 AMD (AMD)"              : "AMD",
    "💻 Intel (INTC)"           : "INTC",
    "🔵 IBM (IBM)"              : "IBM",
    "🛸 Palantir (PLTR)"        : "PLTR",
    "🏠 Airbnb (ABNB)"          : "ABNB",
    "🚀 Uber (UBER)"            : "UBER",
    "📷 Snap (SNAP)"            : "SNAP",
    "🎮 ROBLOX (RBLX)"         : "RBLX",
    "✏️ Other (type manually)"  : "__CUSTOM__",
}

selected_label = st.sidebar.selectbox("Stock Ticker", list(TICKER_OPTIONS.keys()), index=0)
selected_value = TICKER_OPTIONS[selected_label]

if selected_value == "__CUSTOM__":
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper().strip()
else:
    ticker = selected_value

period  = st.sidebar.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=1)
horizon = st.sidebar.slider("Forecast Days (Prophet)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.header("🔑 API Connection Status")

# Fetch API keys — gracefully handles missing secrets.toml (local dev)
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

news_api_key = _get_secret("NEWS_API_KEY")

if not news_api_key:
    st.sidebar.warning("⚠️ NewsAPI key not found. Using mock sentiment data.")
else:
    st.sidebar.success("✅ Connected to NewsAPI")

# ── Data Loading with Retry + Fallback ───────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    Tries yf.Ticker first, then yf.download as fallback.
    Retries up to 3 times with exponential back-off on rate-limit errors.
    """
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        # yf.download returns MultiIndex columns sometimes
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, c)) for c in df.columns]
            rename_map = {c: c.split("_")[0] for c in df.columns if "_" in c}
            df.rename(columns=rename_map, inplace=True)
        df.rename(columns={"index": "Date", "Datetime": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        # Keep only the columns we need (case-insensitive search)
        available = {c.lower(): c for c in df.columns}
        final_cols = []
        for c in cols:
            match = available.get(c.lower())
            if match:
                final_cols.append(match)
        df = df[final_cols].copy()
        df.columns = [c.title() if c.lower() != "date" else "Date" for c in df.columns]
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        return df[cols]

    last_err = None
    for attempt in range(3):
        try:
            # Primary: Ticker.history
            raw = yf.Ticker(ticker).history(period=period)
            if raw.empty:
                raise ValueError(f"No data returned for '{ticker}'.")
            return _clean(raw)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in
                                ["too many requests", "rate limit", "429", "throttle"])
            if is_rate_limit or attempt < 2:
                wait = 2 ** attempt          # 1 s → 2 s → 4 s
                st.toast(f"⏳ Yahoo Finance rate-limited. Retrying in {wait}s… (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                break

    # Fallback: yf.download (different code-path, usually bypasses rate limit)
    try:
        period_map = {"6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = period_map.get(period, 365)
        end   = datetime.today()
        start = end - timedelta(days=days)
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"No data for '{ticker}'.")
        return _clean(raw)
    except Exception as fallback_err:
        raise RuntimeError(
            f"Could not fetch data for **{ticker}** after 3 attempts.\n\n"
            f"Primary error: `{last_err}`\n\nFallback error: `{fallback_err}`\n\n"
            "Yahoo Finance may be temporarily rate-limiting this IP. "
            "Please wait a minute and try again, or check that the ticker symbol is valid."
        ) from fallback_err


@st.cache_data(ttl=3600)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_7"]  = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["Daily_Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────
def mock_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    scores = []
    for _, row in df.iterrows():
        base  = row.get("Daily_Return", 0)
        score = np.clip(base * 3 + np.random.normal(0, 0.2), -1, 1)
        scores.append({"Date": row["Date"], "sentiment": round(float(score), 4)})
    return pd.DataFrame(scores)


@st.cache_data(ttl=3600)
def fetch_news_sentiment(ticker: str, api_key: str, days_back: int = 30):
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
        company_names = {
            "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
            "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "NVIDIA"
        }
        query     = company_names.get(ticker, ticker)
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        articles  = newsapi.get_everything(
            q=query, from_param=from_date, language='en',
            sort_by='publishedAt', page_size=100
        )
        analyzer = SentimentIntensityAnalyzer()
        records  = []
        for a in articles.get('articles', []):
            headline = (a.get('title') or '') + ' ' + (a.get('description') or '')
            score    = analyzer.polarity_scores(headline)
            date     = pd.to_datetime(a['publishedAt']).normalize().tz_localize(None)
            records.append({'Date': date, 'sentiment': score['compound']})
        df = pd.DataFrame(records)
        if df.empty:
            return None
        return df.groupby('Date').mean().reset_index()
    except Exception:
        return None


def get_sentiment(ticker: str, df: pd.DataFrame, news_key: str) -> pd.DataFrame:
    """Return a sentiment DataFrame aligned to df['Date'].
    Uses NewsAPI + VADER; falls back to mock sentiment if key is missing.
    """
    if news_key:
        ndf = fetch_news_sentiment(ticker, news_key)
        if ndf is not None and not ndf.empty:
            merged = pd.merge(df[['Date']], ndf, on='Date', how='left')
            merged['sentiment'] = merged['sentiment'].fillna(0)
            return merged

    return mock_sentiment(df)


# ── Main Data Pipeline ────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker} data from Yahoo Finance…"):
    try:
        raw  = load_data(ticker, period)
        df   = add_indicators(raw)
        sent = get_sentiment(ticker, df, news_api_key)

        # Safe merge — avoids length mismatches from .values assignment
        df = pd.merge(df, sent[["Date", "sentiment"]], on="Date", how="left")
        df["sentiment"] = df["sentiment"].fillna(0)

    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
latest     = df["Close"].iloc[-1]
prev       = df["Close"].iloc[-2]
change     = latest - prev
change_pct = change / prev * 100
avg_sent   = df["sentiment"].tail(30).mean()
rsi_val    = df["RSI"].iloc[-1]

col1.metric("💰 Current Price",        f"${latest:.2f}",    f"{change_pct:+.2f}%")
col2.metric("📊 RSI (14)",             f"{rsi_val:.1f}",
            "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
col3.metric("🗣️ Avg Sentiment (30d)", f"{avg_sent:.3f}",
            "Bullish 🟢" if avg_sent > 0 else "Bearish 🔴")
col4.metric("📈 30d Return",
            f"{((latest / df['Close'].iloc[-30]) - 1) * 100:.2f}%")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Indicators", "🔮 Prophet Forecast",
                                    "🗣️ Sentiment", "📋 Raw Data"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"],
                                  high=df["High"], low=df["Low"],
                                  close=df["Close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_7"],
                              line=dict(color="#00d4ff", width=1.5), name="MA 7"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_21"],
                              line=dict(color="#a29bfe", width=1.5), name="MA 21"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"],
                              line=dict(color="#fdcb6e", width=1, dash="dot"),
                              name="BB Upper"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"],
                              fill="tonexty", line=dict(color="#fdcb6e", width=1, dash="dot"),
                              fillcolor="rgba(253,203,110,0.07)", name="BB Lower"))
    fig.update_layout(template="plotly_dark", height=500,
                       title=f"{ticker} Price Chart",
                       xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        fig_rsi = px.line(df, x="Date", y="RSI", title="RSI (14)",
                           template="plotly_dark", color_discrete_sequence=["#ff6b6b"])
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        st.plotly_chart(fig_rsi, use_container_width=True)
    with col_macd:
        fig_macd = px.line(df, x="Date", y="MACD", title="MACD",
                            template="plotly_dark", color_discrete_sequence=["#a29bfe"])
        st.plotly_chart(fig_macd, use_container_width=True)

with tab2:
    try:
        from prophet import Prophet
        prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        future   = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"],
                                    mode="lines", name="Historical",
                                    line=dict(color="#00d4ff")))
        fig_p.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                    mode="lines", name="Forecast",
                                    line=dict(color="#a29bfe", dash="dash")))
        fig_p.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                                    fill=None, mode="lines",
                                    line=dict(color="#6c5ce7", width=0), showlegend=False))
        fig_p.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                                    fill="tonexty", mode="lines",
                                    line=dict(color="#6c5ce7", width=0),
                                    fillcolor="rgba(108,92,231,0.15)", name="CI Band"))
        fig_p.update_layout(template="plotly_dark", height=500,
                             title=f"{ticker} — {horizon}-Day Prophet Forecast")
        st.plotly_chart(fig_p, use_container_width=True)

        next_pred = forecast["yhat"].iloc[-1]
        ci_low    = forecast["yhat_lower"].iloc[-1]
        ci_high   = forecast["yhat_upper"].iloc[-1]
        st.info(f"📅 **{horizon}-day forecast**: ${next_pred:.2f}  "
                f"(Range: ${ci_low:.2f} – ${ci_high:.2f})")
    except ImportError:
        st.warning("Prophet is not installed. Run `pip install prophet` to enable forecasting.")
    except Exception as e:
        st.error(f"Prophet error: {e}")

with tab3:
    fig_s = px.bar(df, x="Date", y="sentiment",
                    color="sentiment",
                    color_continuous_scale=["#ff6b6b", "#ffeaa7", "#00b894"],
                    title="Daily Sentiment Score", template="plotly_dark")
    st.plotly_chart(fig_s, use_container_width=True)

    rolling_sent = df.set_index("Date")["sentiment"].rolling(7).mean()
    st.line_chart(rolling_sent, use_container_width=True)

with tab4:
    st.dataframe(df.sort_values("Date", ascending=False).head(50),
                  use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Yahoo Finance · Prophet · VADER · Streamlit")
