import json
import requests
import gc
import pandas as pd
from textblob import TextBlob

# Load the configuration file that contains API keys and other settings
with open('config/config.json') as f:
    config = json.load(f)

# Function to fetch news headlines using NewsAPI
def fetch_news(api: str, query: str, page=1) -> list:
    # Construct the API URL with the given query and page number
    url = f"https://newsapi.org/v2/everything?q={query}&page={page}&apiKey={api}"
    response = requests.get(url)
    
    # Extract the headlines from the API response
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles if "title" in article]
    return headlines

# Function to fetch daily stock data from Alpha Vantage API
def get_stock_data(symbol: str, api_key: str):
    endpoint = "https://www.alphavantage.co/query"

    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',
        'datatype': 'json',
        'apikey': api_key
    }

    response = requests.get(endpoint, params)

    # Process the response if successful and return a DataFrame
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            pandasDf = pd.DataFrame.from_dict(time_series, orient='index')
            pandasDf.index = pd.to_datetime(pandasDf.index)  # Convert index to datetime
            pandasDf = pandasDf[['4. close']]  # Keep only the closing price
            pandasDf.rename(columns={'4. close': 'Close'}, inplace=True)
            return pandasDf
    else:
        return None

# Function to analyze sentiment of news headlines using TextBlob
def analyze_sentiment(headlines: list) -> pd.Series:
    sentiments = []

    # Compute the sentiment polarity for each headline
    for headline in headlines:
        sentiment_score = TextBlob(headline).sentiment.polarity
        sentiments.append(sentiment_score)

    # Return the mean sentiment score
    return pd.Series(sentiments).mean()

# Function to generate basic trading signals based on z-score
def generate_signals(data: pd.DataFrame, z_threshold=1.5) -> pd.DataFrame:
    # Initialize signals column
    data['signal'] = 0

    # Generate signals based on z-score thresholds
    data.loc[data['zscore'] > z_threshold, 'signal'] = -1  # Short spread
    data.loc[data['zscore'] < -z_threshold, 'signal'] = 1  # Long spread
    data.loc[data['zscore'].abs() < 0.5, 'signal'] = 0  # No significant deviation

    return data

# Function to calculate the sentiment spread between two assets
def calculate_sentiment_spread(sentiment1, sentiment2):
    return sentiment1 - sentiment2

# Function to merge stock data and sentiment score into a hybrid signal
def merge_signals(stock_data: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
    # Add sentiment score to the DataFrame
    stock_data['Sentiment'] = sentiment_score

    # Combine signal and sentiment into a hybrid signal
    stock_data['Hybrid_Signal'] = 0.5 * stock_data['signal'] + 0.5 * sentiment_score
    return stock_data

# Main script execution
if __name__ == '__main__':
    # Fetch news headlines and analyze sentiment for TSLA
    headlinesTSLA = fetch_news(config['NEWS_API_KEY'], 'TSLA')
    scoreTSLA = analyze_sentiment(headlinesTSLA)
    
    # Fetch news headlines and analyze sentiment for NIO
    headlinesNIO = fetch_news(config['NEWS_API_KEY'], 'NIO')
    scoreNIO = analyze_sentiment(headlinesNIO)

    # Fetch stock data for TSLA and NIO
    dataTSLA = get_stock_data('TSLA', config['ALPHA_VANTAGE_API_KEY'])
    dataNIO = get_stock_data('NIO', config['ALPHA_VANTAGE_API_KEY'])

    # Rename and combine the stock data for both assets
    dataTSLA.rename(columns={'Close': 'CloseTSLA'}, inplace=True)
    dataTSLA['CloseNIO'] = dataNIO['Close']

    # Prepare the final DataFrame for analysis
    finalData = dataTSLA
    del dataTSLA, dataNIO  # Free up memory
    finalData = finalData.iloc[:7]  # Consider the last 7 rows
    gc.collect()

    # Ensure close values are numeric
    finalData['CloseTSLA'] = pd.to_numeric(finalData['CloseTSLA'], errors='coerce')
    finalData['CloseNIO'] = pd.to_numeric(finalData['CloseNIO'], errors='coerce')

    # Calculate spread and z-score
    finalData['spread'] = finalData['CloseTSLA'] - finalData['CloseNIO']
    finalData['zscore'] = (finalData['spread'] - finalData['spread'].mean()) / finalData['spread'].std()

    # Generate trading signals based on z-score
    data = generate_signals(finalData)

    # Merge signals with sentiment spread to generate hybrid signals
    hybridSignal = merge_signals(data, calculate_sentiment_spread(scoreTSLA, scoreNIO))

    # Print the rounded hybrid signal for the most recent date
    print(round(hybridSignal['Hybrid_Signal'].iloc[0]))