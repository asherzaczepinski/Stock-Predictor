import re
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from scipy import stats
from datetime import datetime, timedelta

polygon_api_key = 'POLYGON_API_KEY'
openai_api_key = 'OPENAI_API_KEY'

# Helper function to add a day to given date
def add_day(date):
    return date + timedelta(days=1)

# Fetch article descriptions for any given date
def retrieve_article_descriptions(date, stock_ticker):
    date = date.strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/reference/news?ticker={stock_ticker}&published_utc={date}&apiKey={polygon_api_key}"
    print(f"Fetching article descriptions for date: {date}")
    result = requests.get(url).json()
    print("Successfully fetched article descriptions.")
    return [r.get("description", r.get("title")) for r in result.get("results", [])]

# Get sentiment score
def get_sentiment(description, stock_ticker):
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    url = "https://api.openai.com/v1/engines/text-davinci-003/completions"
    prompt = f"Please analyze the sentiment of the following news about {stock_ticker} stock:\n\n{description}\n\nThe sentiment score ranges from 0 (most negative) to 1 (most positive). A 0.5 score would indicate a neutral sentiment. Return only the sentiment score rounded to two decimal points without any words."
    data = {"prompt": prompt, "max_tokens": 50, "temperature": 0.7}
    print("Analyzing sentiment...")
    response = requests.post(url, headers=headers, json=data, timeout=300).json()
    if 'choices' in response and len(response['choices']) > 0:
        sentiment_score = float(re.search(r"\d+\.\d+", response["choices"][0]["text"].strip()).group())
        print(f"Sentiment analysis result: {sentiment_score}")
        return sentiment_score
    else:
        print(f"Failed to get sentiment for the following description:\n{description}. The response was this {response}")
        return None  # returning None if sentiment analysis failed

# Get stock prices
def get_stock_prices(stock_ticker, start_date, end_date):
    print("Getting stock prices")
    end_date = add_day(end_date)
    # Download data for the stock ticker within the given range
    data = yf.download(stock_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    # Filter out only the 'Close' prices
    close_prices = data['Close']
    # Create a list of lists, where each inner list is [price, date]
    close_prices_list = [[float(price), date.date()] for date, price in close_prices.items()]
    print("Successfully got stock prices")
    print(close_prices_list)
    return close_prices_list

def analyze(stock_ticker, start_date, end_date):
    # Convert the string dates to datetime format
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Generate the list of dates in the specified range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    sentiments = []
    # For each date in the range
    for date in date_range[:-1]:
        date_str = date.strftime('%Y-%m-%d')
        print(f"\nProcessing date: {date_str}")
        # Retrieve the descriptions of the articles for the stock on this date
        descriptions = retrieve_article_descriptions(date, stock_ticker)
        #The number of article
        print(f"Their are {len(descriptions)} article descriptions that have to be processed")
        # Initialize an empty list to store sentiment scores for this date
        sentiment_scores_for_day = []
        # For each description retrieved for this date
        for description in descriptions:
            # Get the sentiment score for the description
            sentiment_score = get_sentiment(description, stock_ticker)
            # If a sentiment score was returned, append it to the list for the day
            if sentiment_score is not None:
                sentiment_scores_for_day.append(sentiment_score)
        # If there were any sentiment scores for the day, calculate their average and append it to the sentiments list
        # If not, append a NaN
        if sentiment_scores_for_day:
            average_sentiment_for_day = np.mean(sentiment_scores_for_day)
            # Also want to include the date in this this
            sentiments.append([average_sentiment_for_day, date.date()])
            print(f"Average sentiment score for {date_str}: {average_sentiment_for_day}")
        else:
            sentiments.append([np.nan, date.date()])
            print(f"No sentiment scores available for {date_str}. Appending NaN.")
    # Retrieve the closing prices for the stock over the specified period
    closing_prices = get_stock_prices(stock_ticker, start_date, end_date)

    return [sentiments, closing_prices]

def alignLists(analyze_results):
    sentiments, closing_prices = analyze_results
    newSentiments = []
    newStockPrices = []

    # Create a dictionary from the closing_prices list for fast lookup
    closing_prices_dict = {date: price for price, date in closing_prices}

    for sentiment, date in sentiments:
        next_day = add_day(date)
        # If there is a closing price for the next day and there are sentiment scores available, append the values
        if next_day in closing_prices_dict and not np.isnan(sentiment):
            newSentiments.append(sentiment)
            newStockPrices.append(closing_prices_dict[next_day])

    return [newSentiments, newStockPrices]

def correlation(sentiments, stockPrices):
    if len(sentiments) < 3:
        print("Insufficient data points for correlation calculation.")
        return None
    # Calculate Spearman's correlation
    corr, _ = stats.spearmanr(sentiments, stockPrices)
    return corr

if __name__ == "__main__":
    stock_ticker = "AAPL"
    start_date = "2022-07-08"
    end_date = "2022-07-31"
    if not stock_ticker:
        stock_ticker = input("Enter the stock ticker: ")
    if not start_date:
        start_date = input("Enter the start date (YYYY-MM-DD): ")
    if not end_date:
        end_date = input("Enter the end date (YYYY-MM-DD): ")
    sentiments, stockPrices = alignLists(analyze(stock_ticker, start_date, end_date))
    print(correlation(sentiments, stockPrices))
