import os
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.utils import dropna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import time
import asyncio
import websockets
import json

# ======================
# PARAMETERS & SETTINGS
# ======================
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon API key
DATA_FOLDER = "saved_data"
RESULTS_FILE = "scalping_results.xlsx"
TICKER = "MARA"
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-12-31"
INITIAL_BALANCE = 1000
CHUNK_DAYS = 30
TIMEFRAME = "minute"

# Stop-Loss
STOP_LOSS_PCT = 0.01  # 1% stop-loss

os.makedirs(DATA_FOLDER, exist_ok=True)


# ======================
# DATA FETCHING FUNCTION
# ======================
def fetch_polygon_data(ticker, from_date, to_date, timeframe="minute"):
    file_path = os.path.join(
        DATA_FOLDER, f"{ticker}_{from_date}_{to_date}_{timeframe}.csv"
    )
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            print(f"Data loaded from {file_path}")
            return data
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    all_data = []

    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timeframe}/{start_date.strftime('%Y-%m-%d')}/{chunk_end_date.strftime('%Y-%m-%d')}?apiKey={API_KEY}"
        print(f"Fetching data from {start_date} to {chunk_end_date}...")
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df["Date"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("Date", inplace=True)
                df.rename(
                    columns={
                        "o": "Open",
                        "h": "High",
                        "l": "Low",
                        "c": "Close",
                        "v": "Volume",
                    },
                    inplace=True,
                )
                all_data.append(df[["Open", "High", "Low", "Close", "Volume"]])
            else:
                print(f"No results found for {start_date} to {chunk_end_date}.")
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.text}")

        start_date = chunk_end_date + timedelta(days=1)
        time.sleep(
            12
        )  # Add a delay to avoid hitting API rate limits (5 requests per minute)

    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        return final_data
    print("No data fetched.")
    return pd.DataFrame()


# ======================
# TECHNICAL INDICATORS
# ======================
def add_indicators(data):
    if data.empty:
        print("Warning: Data is empty after fetching!")
        return data

    data = dropna(data)
    data["EMA5"] = EMAIndicator(data["Close"], window=5).ema_indicator()  # Fast EMA
    data["EMA9"] = EMAIndicator(data["Close"], window=9).ema_indicator()  # Slow EMA
    data["RSI"] = RSIIndicator(data["Close"], window=10).rsi()  # Shorter RSI
    data["VWAP"] = VolumeWeightedAveragePrice(
        data["High"], data["Low"], data["Close"], data["Volume"]
    ).volume_weighted_average_price()  # VWAP for scalping

    # Add Bollinger Bands
    bb = BollingerBands(data["Close"])
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()

    # Add MACD
    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()

    return data.dropna()


# ======================
# TRAINING FUNCTION
# ======================
def train_model(data):
    if data.empty:
        print("Error: No training data available.")
        return None

    # Define features and target
    features = [
        "EMA5",
        "EMA9",
        "RSI",
        "VWAP",
        "BB_High",
        "BB_Low",
        "MACD",
        "MACD_Signal",
    ]
    data["Target"] = np.where(
        data["Close"].shift(-1) > data["Close"], 1, 0
    )  # 1 if price increases, 0 otherwise
    data.dropna(inplace=True)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        data[features], data["Target"], test_size=0.2, random_state=42
    )

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return model


# ======================
# SCALPING STRATEGY WITH STOP-LOSS
# ======================
def scalping_strategy(data, model, initial_balance=INITIAL_BALANCE):
    if data.empty or model is None:
        print("Error: No data or model available for scalping.")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize variables
    balance = initial_balance
    position = 0  # Current position (number of shares held)
    trade_logs = []  # List to store trade details
    stop_loss_price = None  # Stop-loss price for the current position

    # Generate signals using the trained model
    data["Signal"] = model.predict(
        data[
            ["EMA5", "EMA9", "RSI", "VWAP", "BB_High", "BB_Low", "MACD", "MACD_Signal"]
        ]
    )

    # Simulate trades
    for i in range(1, len(data)):
        current_price = data["Close"].iloc[i]

        # Check stop-loss condition
        if position > 0 and current_price <= stop_loss_price:
            # Sell due to stop-loss
            balance += position * current_price
            trade_logs.append(
                {
                    "Date": data.index[i],
                    "Action": "Sell (Stop-Loss)",
                    "Price": current_price,
                    "Shares": position,
                    "Balance": balance,
                }
            )
            position = 0
            stop_loss_price = None

        # Buy signal
        if data["Signal"].iloc[i] == 1 and position == 0:
            buy_price = current_price
            shares_bought = balance // buy_price  # Use all available balance
            if shares_bought > 0:
                position = shares_bought
                balance -= shares_bought * buy_price
                stop_loss_price = buy_price * (1 - STOP_LOSS_PCT)  # Set stop-loss
                trade_logs.append(
                    {
                        "Date": data.index[i],
                        "Action": "Buy",
                        "Price": buy_price,
                        "Shares": shares_bought,
                        "Balance": balance,
                    }
                )

        # Sell signal
        elif data["Signal"].iloc[i] == 0 and position > 0:
            sell_price = current_price
            balance += position * sell_price
            trade_logs.append(
                {
                    "Date": data.index[i],
                    "Action": "Sell",
                    "Price": sell_price,
                    "Shares": position,
                    "Balance": balance,
                }
            )
            position = 0
            stop_loss_price = None

    # Convert trade logs to DataFrame
    trade_logs_df = pd.DataFrame(trade_logs)

    # Calculate summary metrics
    total_return = (balance - initial_balance) / initial_balance
    win_rate = (
        trade_logs_df[trade_logs_df["Action"] == "Sell"]["Price"]
        > trade_logs_df[trade_logs_df["Action"] == "Buy"]["Price"]
    ).mean()
    max_drawdown = (data["Close"].cummax() - data["Close"]).max()

    summary = pd.DataFrame(
        {
            "Total Return": [total_return],
            "Win Rate": [win_rate],
            "Max Drawdown": [max_drawdown],
        }
    )

    return trade_logs_df, summary


# Execution
train_data = add_indicators(
    fetch_polygon_data(TICKER, TRAIN_START_DATE, TRAIN_END_DATE, timeframe=TIMEFRAME)
)
test_data = add_indicators(
    fetch_polygon_data(TICKER, "2024-01-01", "2024-01-10", timeframe=TIMEFRAME)
)

# Train the model
model = train_model(train_data)

# Run scalping strategy with the trained model
if model:
    trade_logs, summary = scalping_strategy(test_data, model)

    # Save trade logs and summary to Excel
    with pd.ExcelWriter(RESULTS_FILE) as writer:
        trade_logs.to_excel(writer, sheet_name="Trade Logs", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Results saved to {RESULTS_FILE}")
else:
    print("Scalping strategy skipped due to model training failure.")
