import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint, adfuller
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.stats import zscore
from dotenv import load_dotenv

load_dotenv()

class QuantTradingSystem:
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.sentiment_model, self.tokenizer = self.load_sentiment_model()

    @staticmethod
    def load_sentiment_model():
        # FinBERT model for sentiment analysis
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        return model, tokenizer

    async def fetch_news(self, session, query):
        # Asynchronously fetch news
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}"
        async with session.get(url) as response:
            data = await response.json()
            return [article["title"] for article in data.get("articles", [])]

    async def get_stock_data(self, session, symbol):
        # Asynchronously fetch stock data
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.alpha_vantage_key,
        }
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                df.index = pd.to_datetime(df.index)
                df = df[["4. close"]].astype(float)
                df.rename(columns={"4. close": symbol}, inplace=True)
                return df
        return None

    def analyze_sentiment(self, headlines):
        # Analyze sentimentt
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = (scores[:, 2] - scores[:, 0]).mean().item()
        return sentiment_score

    def cointegration_test(self, x, y):
        # Johansen cointegration test (needs to have p-value < 0.05)!!!
        score, p_value, _ = coint(x, y)
        return p_value

    def adf_test(self, series):
        # Stationarity test (needs to have p-value < 0.05)!!!
        result = adfuller(series)
        return result[1]

    def pca_factor_analysis(self, data):
        # Principal Component Analysis to extract factors from stock data
        pca = PCA(n_components=1)
        principal_component = pca.fit_transform(data)
        return pd.Series(principal_component.flatten(), index=data.index)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            news_tsla, news_nio, data_tsla, data_nio = await asyncio.gather(
                self.fetch_news(session, "TSLA"),
                self.fetch_news(session, "NIO"),
                self.get_stock_data(session, "TSLA"),
                self.get_stock_data(session, "NIO"),
            )

        # Sentiment analysis
        score_tsla = self.analyze_sentiment(news_tsla)
        score_nio = self.analyze_sentiment(news_nio)

        # Merge stock data
        data = data_tsla.join(data_nio, how="inner")
        data["spread"] = data["TSLA"] - data["NIO"]

        # Statistical validation
        cointegration_p = self.cointegration_test(data["TSLA"], data["NIO"])
        adf_p = self.adf_test(data["spread"])
        print(f"Cointegration: {cointegration_p:.5f} | ADF: {adf_p:.5f}")

        if cointegration_p < 0.05 and adf_p < 0.05:
            print("Valid pair for mean-reversion trading.")
        else:
            print("No strong statistical evidene for mean-reversion")

        # PCA Factor Extraction
        data["factor"] = self.pca_factor_analysis(data[["TSLA", "NIO"]])

        # Generate Z-score and trading signals
        data["zscore"] = zscore(data["spread"])
        data["signal"] = np.where(data["zscore"] > 1.5, -1, 0)  # Short
        data["signal"] = np.where(data["zscore"] < -1.5, 1, data["signal"])  # Long

        # Save for backtesting
        data[["TSLA", "NIO", "spread", "zscore", "signal", "factor"]].to_csv("../data/signals.csv")

        print("Statistical validation + signals saved.")

if __name__ == "__main__":
    trading_system = QuantTradingSystem()
    asyncio.run(trading_system.run())