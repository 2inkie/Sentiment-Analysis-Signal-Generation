import json
import requests
import pandas as pd
from textblob import TextBlob

with open('../config/config.json') as f:
    config = json.load(f)

def fetch_news(api, query, page=1):
    url = f"https://newsapi.org/v2/everything?q={query}&page={page}&apiKey={api}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles if "title" in article]
    return headlines

def analyze_sentiment(headlines):
    sentiments = []

    for headline in headlines:
        sentiment_score = TextBlob(headline).sentiment.polarity
        sentiments.append(sentiment_score)

    return pd.Series(sentiments).mean()

if __name__ == '__main__':
    headlines = fetch_news(config['NEWS_API_KEY'], 'TSLA')
    score = analyze_sentiment(headlines)

    # print(headlines)
    print(score)