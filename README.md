# 📈 Stock Price Prediction with Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Model](https://img.shields.io/badge/Model-Prophet-purple?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20NewsAPI-yellow?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20NewsAPI-informational?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> An interactive stock price analysis and forecasting dashboard combining **Prophet time-series forecasting**, **technical indicators**, and **real-time news sentiment analysis** via NewsAPI + VADER — deployed as a Streamlit web application.

---

## 📌 Problem Statement

Stock prices are influenced by two forces: **historical price patterns** and **market sentiment**. This project fuses both into one interactive dashboard:

- **Quantitative signal:** Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages) computed on OHLCV data from Yahoo Finance
- **Sentiment signal:** VADER sentiment scores applied to live financial news headlines from NewsAPI
- **Forecast:** Prophet time-series model producing a configurable day-ahead price forecast with confidence intervals

---

## 🎯 Features

| Feature | Details |
|---|---|
| 📊 Live Price Data | Fetches OHLCV data from Yahoo Finance with retry + fallback logic |
| 📈 Technical Indicators | RSI (14), MACD, Bollinger Bands, MA 7 & MA 21 |
| 🔮 Prophet Forecast | Configurable 7–90 day forecast with uncertainty bands |
| 🗣️ Sentiment Analysis | NewsAPI headlines → VADER compound score per day |
| 🏢 30 Stock Tickers | Dropdown of popular companies + custom ticker input |
| 🌑 Dark UI | Dark-themed Plotly charts with candlestick, RSI, MACD views |
| 💾 Caching | 1-hour Streamlit cache on all data/model calls |

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────┐
                    │        DATA SOURCES          │
                    └─────────────────────────────┘
                           │              │
              ┌────────────┘              └────────────┐
              ▼                                        ▼
   Yahoo Finance (yfinance)                     NewsAPI
   ├── OHLCV daily data                  ├── Financial news headlines
   ├── 6mo / 1y / 2y / 5y history       ├── Last 30 days
   └── Retry + fallback on rate limits   └── Company name query
              │                                        │
              ▼                                        ▼
   Technical Indicators                    Sentiment Pipeline
   ├── MA 7 / MA 21                        ├── VADER polarity scoring
   ├── RSI (14-day)                        ├── Daily aggregation
   ├── MACD (EMA 12/26)                   └── Date-aligned merge
   └── Bollinger Bands (20-day)
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
                   Feature DataFrame
                   [OHLCV | Indicators | Sentiment Score]
                             │
                             ▼
                   Prophet Forecasting
                   ├── Weekly + Yearly seasonality
                   ├── Configurable forecast horizon (7–90 days)
                   └── yhat + confidence interval (yhat_upper / yhat_lower)
                             │
                             ▼
                   Streamlit Dashboard
                   ├── Ticker dropdown (29 companies + custom)
                   ├── Candlestick + indicator charts
                   ├── Prophet forecast chart
                   ├── Sentiment bar chart + 7-day rolling average
                   └── Raw data table
```

---

## 📊 Dashboard Tabs

### 📈 Price & Indicators
- **Candlestick chart** with MA 7, MA 21, and Bollinger Bands overlay
- **RSI (14)** with overbought (70) / oversold (30) reference lines
- **MACD** line chart

### 🔮 Prophet Forecast
- Historical price overlaid with Prophet's forecast line
- Shaded confidence interval band
- Forecast summary: predicted price + range for the selected horizon

### 🗣️ Sentiment
- Daily bar chart of VADER compound scores (–1 to +1), color-coded red → yellow → green
- 7-day rolling average sentiment line chart

### 📋 Raw Data
- Full DataFrame with all computed columns, sorted by most recent date

---

## 🧠 Key Technical Decisions

**Why VADER over FinBERT?**
VADER is lightweight, runs entirely offline, and requires no GPU or API quota. For a live dashboard that processes 100 headlines per request, it gives near-instant results. FinBERT would add meaningful accuracy but requires transformer inference — not suitable for a free-tier deployment.

**Why NewsAPI over Reddit (PRAW)?**
NewsAPI provides structured, clean financial headlines with timestamps via a simple REST call. Reddit scraping via PRAW requires OAuth credentials and is rate-limited more aggressively; it also returns noisier, less finance-specific content.

**Why Prophet over LSTM?**
Prophet runs in seconds without GPU, handles missing data and seasonality automatically, and requires no data normalization or sequence windowing. For a dashboard where a user wants a forecast on any of 29 tickers instantly, Prophet is the right tool.

**Why retry + fallback for Yahoo Finance?**
Yahoo Finance's free API is rate-limited per IP. The app uses `yf.Ticker().history()` first, retries 3× with exponential back-off (1s → 2s → 4s), then falls back to `yf.download()` — a different code path that often bypasses the rate limit.

**Why mock sentiment as fallback?**
If the NewsAPI key is missing or the request fails, the app generates deterministic mock sentiment scores derived from the stock's own daily returns + controlled noise. This keeps the dashboard fully functional for demo purposes without any API key.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### API Key Setup
Create `.streamlit/secrets.toml` in the project root:
```toml
NEWS_API_KEY = "your_newsapi_key"   # free tier at newsapi.org
```

Or set it as an environment variable:
```bash
export NEWS_API_KEY=your_newsapi_key
```

> **No API key?** The dashboard still works — it uses mock sentiment data derived from price returns as a fallback.

### Run the Dashboard
```bash
git clone https://github.com/charansai-1411/Stock-Price-Prediction-with-Sentiment-Analysis
cd Stock-Price-Prediction-with-Sentiment-Analysis
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
Stock-Price-Prediction-with-Sentiment-Analysis/
├── app.py                        # Streamlit dashboard (single-file app)
├── Stock_Price_Analysis.ipynb    # Exploratory ML pipeline notebook
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── secrets.toml              # API keys (not committed to Git)
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Stock OHLCV data (Yahoo Finance) |
| `NewsAPI` (`newsapi-python`) | Financial news headlines |
| `vaderSentiment` | Lexicon-based sentiment scoring |
| `Prophet` | Time-series price forecasting |
| `Streamlit` | Interactive web dashboard |
| `Plotly` | Interactive charts (candlestick, RSI, MACD) |
| `Pandas / NumPy` | Data manipulation & indicator computation |

---

## ⚙️ Configuration Options

| Sidebar Control | Options | Default |
|---|---|---|
| Stock Ticker | Dropdown of 29 companies + custom input | Apple (AAPL) |
| History Period | 6mo / 1y / 2y / 5y | 1y |
| Forecast Days | 7 – 90 days (slider) | 30 |

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not financial advice. Stock price prediction is inherently uncertain — do not make investment decisions based on model outputs.

---

## 👤 Author

**Y. Charan Sai**  
BE — Artificial Intelligence & Data Science  
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/charan-sai-8b0a42283)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/charansai-1411)
