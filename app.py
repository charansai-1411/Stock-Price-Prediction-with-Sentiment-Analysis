import os
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import ta

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

# Sidebar
st.sidebar.header("⚙️ Configuration")
ticker  = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period  = st.sidebar.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=1)
horizon = st.sidebar.slider("Forecast Days (Prophet)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.header("🔑 API Connection Status")

# Fetch API keys from Streamlit secrets or OS environment variables
news_api_key = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
reddit_client_id = st.secrets.get("REDDIT_CLIENT_ID", os.getenv("REDDIT_CLIENT_ID", ""))
reddit_client_secret = st.secrets.get("REDDIT_CLIENT_SECRET", os.getenv("REDDIT_CLIENT_SECRET", ""))

if not news_api_key or not reddit_client_id:
    st.sidebar.warning("⚠️ Live API keys not found in environment. Using mock sentiment data.")
else:
    st.sidebar.success("✅ Connected to live Sentiment APIs")

@st.cache_data(ttl=3600)
def load_data(ticker, period):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    df = yf.Ticker(ticker, session=session).history(period=period)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]

@st.cache_data
def add_indicators(df):
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
    return df

def mock_sentiment(df):
    np.random.seed(42)
    scores = []
    for _, row in df.iterrows():
        base  = row.get("Daily_Return", 0)
        score = np.clip(base * 3 + np.random.normal(0, 0.2), -1, 1)
        scores.append({"Date": row["Date"], "sentiment": round(score, 4)})
    return pd.DataFrame(scores)

@st.cache_data(ttl=3600)
def fetch_news_sentiment(ticker, api_key, days_back=30):
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=api_key)
        company_names = {"AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon", "NVDA": "NVIDIA"}
        query = company_names.get(ticker, ticker)
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=query, from_param=from_date, language='en', sort_by='publishedAt', page_size=100)
        
        analyzer = SentimentIntensityAnalyzer()
        records = []
        for a in articles.get('articles', []):
            headline = (a.get('title') or '') + ' ' + (a.get('description') or '')
            score = analyzer.polarity_scores(headline)
            date = pd.to_datetime(a['publishedAt']).normalize().tz_localize(None)
            records.append({'Date': date, 'sentiment': score['compound']})
            
        df = pd.DataFrame(records)
        if df.empty: return None
        return df.groupby('Date').mean().reset_index()
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_reddit_sentiment(ticker, client_id, client_secret, user_agent="stock_sentiment_dashboard"):
    try:
        import praw
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        
        analyzer = SentimentIntensityAnalyzer()
        records = []
        subreddit = reddit.subreddit('stocks+investing+wallstreetbets')
        for submission in subreddit.search(ticker, time_filter='month', limit=100):
            text = (submission.title or '') + " " + (submission.selftext or '')
            score = analyzer.polarity_scores(text)
            date = pd.to_datetime(submission.created_utc, unit='s').normalize().tz_localize(None)
            records.append({'Date': date, 'sentiment': score['compound']})
            
        df = pd.DataFrame(records)
        if df.empty: return None
        return df.groupby('Date').mean().reset_index()
    except Exception as e:
        return None

def get_sentiment(ticker, df, news_key, red_id, red_sec):
    dfs = []
    if news_key:
        ndf = fetch_news_sentiment(ticker, news_key)
        if ndf is not None and not ndf.empty: dfs.append(ndf)
    if red_id and red_sec:
        rdf = fetch_reddit_sentiment(ticker, red_id, red_sec)
        if rdf is not None and not rdf.empty: dfs.append(rdf)
        
    if dfs:
        combined = pd.concat(dfs).groupby('Date').mean().reset_index()
        temp = pd.merge(df[['Date']], combined, on='Date', how='left')
        temp['sentiment'] = temp['sentiment'].fillna(0)
        return temp
    return mock_sentiment(df)

with st.spinner(f"Loading {ticker} data..."):
    try:
        raw  = load_data(ticker, period)
        df   = add_indicators(raw)
        sent = get_sentiment(ticker, df, news_api_key, reddit_client_id, reddit_client_secret)
        df["sentiment"] = sent["sentiment"].values
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# KPI Row
col1, col2, col3, col4 = st.columns(4)
latest       = df["Close"].iloc[-1]
prev         = df["Close"].iloc[-2]
change       = latest - prev
change_pct   = change / prev * 100
avg_sent     = df["sentiment"].tail(30).mean()
rsi_val      = df["RSI"].iloc[-1]

col1.metric("💰 Current Price", f"${latest:.2f}", f"{change_pct:+.2f}%")
col2.metric("📊 RSI (14)", f"{rsi_val:.1f}",
            "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
col3.metric("🗣️ Avg Sentiment (30d)", f"{avg_sent:.3f}",
            "Bullish 🟢" if avg_sent > 0 else "Bearish 🔴")
col4.metric("📈 30d Return",
            f"{((latest / df['Close'].iloc[-30]) - 1) * 100:.2f}%")

st.markdown("---")

# Price Chart with Indicators
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
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
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
