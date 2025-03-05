from common_imports import *

def fetch_historical_data(ticker):
    # Today's date
    end_date = datetime.now()
    # Calculate start date (2 months ago)
    start_date = end_date - timedelta(days=55)

    # Format the start date as a string
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Fetch historical data using yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date_str, interval="15m")
    
    # Handle empty data
    if df.empty:
        print(f"No data found for {ticker}")
        return None
    
    return df


def fetch_historical_order_book(ticker):
    df = fetch_historical_data(ticker)
    if df is not None:
        df["Mid Price"] = (df["High"] + df["Low"]) / 2
        df["Bid Price"] = df["Low"]
        df["Ask Price"] = df["High"]
        df["Bid Volume"] = df["Volume"] * 0.4
        df["Ask Volume"] = df["Volume"] * 0.6
        df["Spread"] = df["Ask Price"] - df["Bid Price"]
    return df