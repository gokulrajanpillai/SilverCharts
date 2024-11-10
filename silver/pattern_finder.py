import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filename, sequence_length=60, future_steps=10):
    """
    Loads the data from CSV, processes it, and prepares the features and labels.
    """
    # Load data and parse dates
    data = pd.read_csv(filename, index_col=0, parse_dates=True, date_format='%Y-%m-%d', on_bad_lines='skip')

    # Debugging: Check the columns and types to find the non-numeric columns
    print("Columns in the data:", data.columns)

    # Strip any spaces in column names
    data.columns = data.columns.str.strip()

    # Check if the necessary columns are present
    required_columns = ['Close', 'RSI', 'Bollinger_Middle', 'Bollinger_Upper', 'Bollinger_Lower', 'CCI']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        # Handle missing columns as appropriate (e.g., raise an error or skip processing)
        return None, None, None, None
    
    # Keep only the numeric columns that are required
    data = data[required_columns]

    # Check for NaN values and handle them
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid values become NaN
    data = data.dropna()  # Drop rows with NaN values after coercion

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences (X) and labels (y)
    X, y = create_sequences_and_labels(scaled_data, sequence_length, future_steps)

    return X, y, scaler, data

def create_sequences_and_labels(data, sequence_length, future_steps):
    """
    Creates sequences of data points (X) and labels (y) for prediction.
    """
    X = []
    y = []
    for i in range(len(data) - sequence_length - future_steps):
        X.append(data[i:i+sequence_length])  # Input sequence
        future_price = data[i+sequence_length+future_steps-1, 0]  # Future price (Close)
        y.append(future_price)  # This can be adjusted depending on the prediction (e.g., price or direction)
    
    return np.array(X), np.array(y)

def interpret_patterns(data, predictions, scaler, original_data, sequence_length):
    """
    This function interprets the patterns from the predicted data and highlights them.
    """
    # Only use numeric columns for inverse transformation
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Inverse transform the predictions and data to original scale
    try:
        price_data = scaler.inverse_transform(numeric_data)
        print(f"Price Data after inverse transformation: {price_data[:5]}")  # Check the first few values
    except ValueError as e:
        print(f"Error during inverse transformation: {e}")

    # Create a new DataFrame for highlighting the patterns
    highlighted_data = original_data.copy()

    # Simple pattern: If the predicted price goes up, flag it as an "Upward Trend"
    # You can customize this with more complex logic based on your model
    trend_column = []
    for i in range(len(predictions) - 1):
        if predictions[i + 1] > predictions[i]:
            trend_column.append('Upward Trend')
        else:
            trend_column.append('Downward Trend')

    # Append a "No Pattern" flag for the last entry
    trend_column.append('No Pattern')
    
    # Ensure that the trend_column matches the length from `sequence_length`
    highlighted_data['Pattern'] = 'No Pattern'  # Default to 'No Pattern'
    
    # Use iloc for positional indexing
    highlighted_data.iloc[sequence_length:len(trend_column) + sequence_length, highlighted_data.columns.get_loc('Pattern')] = trend_column  # Apply the trends to the data

    # Save the highlighted data to a CSV file
    highlighted_data.to_csv("highlighted_patterns.csv")

def main(filename):
    """
    Main function to load data, train model, and interpret patterns.
    """
    sequence_length = 60
    future_steps = 10
    
    # Load and preprocess data
    X, y, scaler, original_data = load_and_preprocess_data(filename, sequence_length, future_steps)
    
    if X is None or y is None:
        print("Error: Data preprocessing failed due to missing columns.")
        return
    
    # Simulate predictions (for demonstration purposes)
    predictions = np.random.rand(len(y))  # Replace with your actual predictions

    # Interpret the patterns and save them to CSV
    interpret_patterns(pd.DataFrame(X.reshape(-1, X.shape[2])), predictions, scaler, original_data, sequence_length)

if __name__ == "__main__":
    # Define your filename (path to the CSV file)
    filename = "silver_data.csv"  # Update with your actual path
    main(filename)
