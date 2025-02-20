import os
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.utils import dropna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
TICKER = "MARA"  # Replace with a valid ticker symbol
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-12-31"
INITIAL_BALANCE = 1000  # Starting balance
CHUNK_DAYS = 30  # Fetch intraday data in 30-day chunks
TIMEFRAME = "minute"  # Use 1-minute data for scalping

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
    return data.dropna()


# ======================
# TRAINING FUNCTION
# ======================
def train_model(data):
    if data.empty:
        print("Error: No training data available.")
        return None

    # Define features and target
    features = ["EMA5", "EMA9", "RSI", "VWAP"]
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
# REAL-TIME TRADING WITH WEBSOCKET
# ======================
async def real_time_trading(model, ticker):
    print("Checking for real-time trading signals...")

    # Initialize variables
    balance = INITIAL_BALANCE
    position = 0  # Current position (number of shares held)
    trade_logs = []  # List to store trade details
    data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Connect to Polygon WebSocket
    async with websockets.connect("wss://socket.polygon.io/stocks") as websocket:
        # Authenticate with API key
        await websocket.send(json.dumps({"action": "auth", "params": API_KEY}))
        # Subscribe to the ticker
        await websocket.send(
            json.dumps({"action": "subscribe", "params": f"A.{ticker}"})
        )

        while True:
            message = await websocket.recv()
            data_json = json.loads(message)

            # Process real-time data
            if data_json[0]["ev"] == "A":  # Aggregate (minute) data
                new_data = {
                    "Date": datetime.fromtimestamp(data_json[0]["t"] / 1000),
                    "Open": data_json[0]["o"],
                    "High": data_json[0]["h"],
                    "Low": data_json[0]["l"],
                    "Close": data_json[0]["c"],
                    "Volume": data_json[0]["v"],
                }
                data = data.append(new_data, ignore_index=True)

                # Add indicators
                data = add_indicators(data)

                # Generate signal using the trained model
                if len(data) >= 10:  # Ensure enough data for indicators
                    signal = model.predict(
                        data[["EMA5", "EMA9", "RSI", "VWAP"]].iloc[[-1]]
                    )

                    # Execute trades
                    if signal == 1 and position == 0:  # Buy signal
                        buy_price = data["Close"].iloc[-1]
                        shares_bought = (
                            balance // buy_price
                        )  # Use all available balance
                        if shares_bought > 0:
                            position = shares_bought
                            balance -= shares_bought * buy_price
                            trade_logs.append(
                                {
                                    "Date": data["Date"].iloc[-1],
                                    "Action": "Buy",
                                    "Price": buy_price,
                                    "Shares": shares_bought,
                                    "Balance": balance,
                                }
                            )
                            print(
                                f"Buy {shares_bought} shares at {buy_price}. Balance: {balance}"
                            )
                    elif signal == 0 and position > 0:  # Sell signal
                        sell_price = data["Close"].iloc[-1]
                        balance += position * sell_price
                        trade_logs.append(
                            {
                                "Date": data["Date"].iloc[-1],
                                "Action": "Sell",
                                "Price": sell_price,
                                "Shares": position,
                                "Balance": balance,
                            }
                        )
                        print(
                            f"Sell {position} shares at {sell_price}. Balance: {balance}"
                        )
                        position = 0  # Reset position after selling


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Fetch and preprocess training data
    train_data = add_indicators(
        fetch_polygon_data(
            TICKER, TRAIN_START_DATE, TRAIN_END_DATE, timeframe=TIMEFRAME
        )
    )

    # Train the model
    model = train_model(train_data)

    if model:
        # Start real-time trading
        print("Starting real-time trading...")
        asyncio.run(real_time_trading(model, TICKER))
    else:
        print("Real-time trading skipped due to model training failure.")
