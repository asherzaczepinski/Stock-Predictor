# Stock-Predictor
ChatGPT Sentiment Analysis on Stock Article Descriptions Correlated with Returns

Runs a correlation between the average sentiment of articles regarding a ticker from any day with the closing price of the ticker the following day. After entering in a span of days it determines if the sentiment of articles published affects the stock price. I use the prompt: 

"Please analyze the sentiment of the following news about {stock_ticker} stock:\n\n{description}\n\nThe sentiment score ranges from 0 (most negative) to 1 (most positive). A 0.5 score would indicate a neutral sentiment. Return only the sentiment score rounded to two decimal points without any words." 

In the future I would like to have it rate the sentiment as good or bad for any particular ticker and track my data off that. 
