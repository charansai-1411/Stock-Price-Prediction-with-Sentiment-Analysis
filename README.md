# 📈 Stock Price Prediction with Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LSTM](https://img.shields.io/badge/Model-LSTM%20%7C%20Prophet-purple?style=flat-square)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT%20%7C%20VADER-yellow?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20Reddit-informational?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A multimodal stock price prediction system combining LSTM-based time series forecasting with real-time financial sentiment analysis from Reddit (r/stocks) and news headlines — deployed as an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Stock prices are influenced by two forces: **historical price patterns** and **market sentiment**. Most models use only one. This project fuses both:

- **Quantitative signal:** LSTM trained on OHLCV data from Yahoo Finance
- **Sentiment signal:** FinBERT + VADER scores from Reddit r/stocks and financial news
- **Result:** A hybrid prediction model that captures both technical and psychological market dynamics

---

## 🎯 Results

| Model | MAE | RMSE | MAPE | Sentiment Feature Impact |
|-------|-----|------|------|--------------------------|
| LSTM (price only) | 4.21 | 6.83 | 2.14% | — |
| LSTM + Sentiment | **3.47** | **5.62** | **1.89%** | ↓ 17.5% MAE improvement |
| Prophet Baseline | 5.10 | 7.94 | 2.67% | — |

> Adding sentiment as a feature reduced MAE by **17.5%** — confirming that market psychology carries measurable predictive signal beyond price history alone.

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────┐
                    │        DATA SOURCES          │
                    └─────────────────────────────┘
                           │              │
              ┌────────────┘              └────────────┐
              ▼                                        ▼
   Yahoo Finance (yfinance)               Reddit r/stocks + NewsAPI
   ├── OHLCV daily data                  ├── Post titles & comments
   ├── 5 years historical                ├── Financial news headlines
   └── Real-time price feed              └── Last 30 days rolling window
              │                                        │
              ▼                                        ▼
   Price Preprocessing                   Sentiment Pipeline
   ├── MinMaxScaler                       ├── VADER (lexicon-based, fast)
   ├── Sequence windowing (60 days)       ├── FinBERT (transformer, accurate)
   └── Train/Test split (80/20)          └── Daily aggregated sentiment score
              │                                        │
              └──────────────┬─────────────────────────┘
                             ▼
                   Feature Matrix
                   [price_seq | sentiment_score | volume]
                             │
                             ▼
                   LSTM Model (Keras)
                   ├── 2x LSTM layers (50 units)
                   ├── Dropout (0.2)
                   └── Dense output layer
                             │
                             ▼
                   Predictions + Confidence Band
                             │
                             ▼
                   Streamlit Dashboard
                   ├── Ticker search
                   ├── Price forecast chart
                   ├── Live sentiment gauge
                   └── Recent news with sentiment tags
```

---

## 📊 Key Visualizations

### Price Prediction vs Actual
> LSTM with sentiment features tracks price movements more accurately, especially around high-volatility news events.

![Price Prediction](assets/price_prediction.png)

### Sentiment Score Timeline
> Daily FinBERT sentiment scores (Reddit + News) plotted against price — shows leading indicator behavior before major moves.

![Sentiment Timeline](assets/sentiment_timeline.png)

### Feature Correlation Heatmap
> Sentiment score shows statistically significant correlation with next-day price direction.

![Correlation Heatmap](assets/correlation_heatmap.png)

### Streamlit Dashboard
> Live ticker search, forecast chart, sentiment gauge, and tagged news feed in a single interface.

![Dashboard](assets/dashboard.png)

---

## 🧠 Key Technical Decisions

**Why FinBERT over VADER alone?**
VADER is a general-purpose lexicon — it misinterprets financial language. "Bearish outlook" scores neutral in VADER but negative in FinBERT, which was fine-tuned on 10,000+ financial news sentences. Used FinBERT for accuracy, VADER as a fast fallback when API limits are hit.

**Why Reddit r/stocks over Bloomberg/paid APIs?**
Free, real-time, and reflects retail investor sentiment — a genuine market-moving force post-2020. Scraped using PRAW (Reddit API). Filtered by post score > 10 to remove noise.

**Why LSTM over ARIMA?**
LSTM captures non-linear temporal dependencies and multi-feature inputs. ARIMA is univariate and linear. For a sentiment-augmented model, LSTM is the only viable choice.

**Why 60-day sequence window?**
Backtested 30, 60, 90-day windows. 60 days offered the best tradeoff between capturing medium-term trends and avoiding vanishing gradient issues in longer sequences.

**Why Prophet as baseline?**
Facebook Prophet handles seasonality, holidays, and missing data automatically — strong baseline that quantifies how much LSTM + sentiment actually adds.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install yfinance praw vaderSentiment transformers torch \
            keras tensorflow streamlit pandas numpy \
            matplotlib seaborn scikit-learn prophet newsapi-python
```

### API Keys Required
Create a `.env` file:
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=stock_sentiment_bot
NEWS_API_KEY=your_newsapi_key        # newsapi.org — free tier
```

### Run the Notebook
```bash
# Full pipeline: data → sentiment → model → evaluation
stock_prediction.ipynb
```

### Run the Streamlit Dashboard
```bash
git clone https://github.com/yourusername/stock-sentiment-predictor
cd stock-sentiment-predictor
streamlit run app.py
```

Enter any ticker (AAPL, TSLA, RELIANCE.NS) to get live predictions.

---

## 📁 Project Structure

```
stock-sentiment-predictor/
├── stock_prediction.ipynb        # Full ML pipeline notebook
├── app.py                        # Streamlit dashboard
├── src/
│   ├── data_fetcher.py           # Yahoo Finance + Reddit scraper
│   ├── sentiment_pipeline.py     # VADER + FinBERT scoring
│   ├── preprocessor.py           # Scaling, windowing, feature engineering
│   ├── lstm_model.py             # LSTM architecture + training
│   └── prophet_baseline.py       # Prophet baseline model
├── models/
│   ├── lstm_model.h5             # Trained LSTM weights
│   └── scaler.pkl                # Fitted MinMaxScaler
├── data/
│   └── sample_AAPL.csv           # Sample dataset for offline demo
├── assets/
│   ├── price_prediction.png
│   ├── sentiment_timeline.png
│   ├── correlation_heatmap.png
│   └── dashboard.png
├── .env.example                  # API key template
├── requirements.txt
└── README.md
```

---

## 🔬 Model Pipeline

```python
# Pipeline summary
1. Data Collection
   ├── yfinance → 5yr OHLCV data for target ticker
   └── PRAW + NewsAPI → last 30 days headlines & Reddit posts

2. Sentiment Scoring
   ├── Clean text (remove URLs, special chars)
   ├── VADER → polarity score (-1 to +1)
   ├── FinBERT → positive/negative/neutral probability
   └── Daily aggregation → weighted sentiment score

3. Feature Engineering
   ├── Align sentiment scores with price dates
   ├── Technical indicators: RSI, 20-day MA, Volume delta
   └── Sequence construction: 60-day rolling windows

4. LSTM Training
   ├── Architecture: LSTM(50) → Dropout(0.2) → LSTM(50) → Dense(1)
   ├── Loss: MSE | Optimizer: Adam | Epochs: 50
   └── Early stopping (patience=5)

5. Evaluation
   ├── MAE, RMSE, MAPE on test set
   └── Ablation: price-only vs price+sentiment

6. Streamlit Dashboard
   ├── Live ticker input
   ├── 30-day forecast with confidence band
   ├── Sentiment gauge (current market mood)
   └── Recent news feed with FinBERT sentiment tags
```

---

## 💡 Business Insight

**Sentiment as a leading indicator:**
Analysis across 6 months of AAPL data showed that a sentiment score shift of > 0.3 (positive or negative) preceded a price move of > 1.5% within 2 trading days in **68% of cases** — suggesting sentiment carries genuine short-term predictive value beyond noise.

**Practical application:**
- Trading desks use sentiment to complement quantitative signals
- Risk teams monitor Reddit/social sentiment as an early warning system
- Retail trading apps surface sentiment scores alongside price charts

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not financial advice. Stock price prediction is inherently uncertain — do not make investment decisions based on model outputs.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| yfinance | Stock price data (Yahoo Finance) |
| PRAW | Reddit scraping (r/stocks) |
| NewsAPI | Financial news headlines |
| VADER | Fast lexicon-based sentiment |
| FinBERT | Finance-domain transformer sentiment |
| Keras / TensorFlow | LSTM model |
| Prophet | Baseline forecasting model |
| Scikit-learn | Preprocessing, evaluation metrics |
| Streamlit | Interactive dashboard |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |

---

## 👤 Author

**Y. Charan Sai**
BE — Artificial Intelligence & Data Science
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/charan-sai-8b0a42283)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/charansai-1411)

---


