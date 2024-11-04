import yfinance as yf
import pandas as pd

# Fetch silver futures data for the past year
silver_data = yf.download("SI=F", period="1y", interval="1h")
silver_data.dropna(inplace=True)  # Remove any missing values

# Convert index to Eastern Time directly
silver_data.index = silver_data.index.tz_convert('US/Eastern')

# Filter for Sunday night data (5 p.m. to midnight ET)
sunday_night_data = silver_data[(silver_data.index.weekday == 6) & (silver_data.index.hour >= 17)]

# Find the peak price for each Sunday night
sunday_peaks = sunday_night_data.groupby(sunday_night_data.index.date)['High'].max()

# Prepare a DataFrame to hold results
results = pd.DataFrame(columns=["Date", "Peak Price", "Values Before", "Values After"])

# Loop through peaks to find values before and after each peak
for peak_date in sunday_peaks.index:
    # Get the peak price and corresponding row
    peak_price = sunday_peaks[peak_date]
    
    # Find the index of the peak
    peak_index = sunday_night_data[sunday_night_data.index.date == peak_date][sunday_night_data['High'] == peak_price].index[0]

    # Get values before and after the peak
    before_values = sunday_night_data.loc[peak_index - pd.DateOffset(hours=1*3): peak_index - pd.DateOffset(hours=1)].values.flatten().tolist()
    after_values = sunday_night_data.loc[peak_index + pd.DateOffset(hours=1): peak_index + pd.DateOffset(hours=1*3)].values.flatten().tolist()

    # Append results to the DataFrame
    results = results.append({
        "Date": peak_date,
        "Peak Price": peak_price,
        "Values Before": before_values,
        "Values After": after_values
    }, ignore_index=True)

# Display the results
print("Sunday night peaks and surrounding values in the past year:")
print(results)
