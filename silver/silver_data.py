import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import CCIIndicator

def fetch_silver_data(interval='1h', period='1y', output_file='silver_data.csv'):
    # Fetch data
    silver_data = yf.download('SI=F', interval=interval, period=period)
    
    # Check if data was fetched successfully
    if silver_data.empty:
        print("No data found for silver.")
        return None

    # Convert to 1-dimensional Series if needed
    close = silver_data['Close'].squeeze()
    high = silver_data['High'].squeeze()
    low = silver_data['Low'].squeeze()

    # Calculate RSI
    rsi = RSIIndicator(close=close, window=14)
    silver_data['RSI'] = rsi.rsi()

    # Calculate Bollinger Bands
    bollinger = BollingerBands(close=close, window=20, window_dev=2)
    silver_data['Bollinger_Middle'] = bollinger.bollinger_mavg()
    silver_data['Bollinger_Upper'] = bollinger.bollinger_hband()
    silver_data['Bollinger_Lower'] = bollinger.bollinger_lband()

    # Calculate CCI
    cci = CCIIndicator(high=high, low=low, close=close, window=20)
    silver_data['CCI'] = cci.cci()

    # Save the DataFrame to a CSV file
    silver_data.to_csv(output_file)
    print(f"Data saved to {output_file}")

    return silver_data

# Fetch and save the data to 'silver_data.csv'
silver_data = fetch_silver_data()
if silver_data is not None:
    print(silver_data.tail())
