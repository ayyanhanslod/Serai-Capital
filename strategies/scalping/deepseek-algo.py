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

# ======================
# PARAMETERS & SETTINGS
# ======================
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon API key
DATA_FOLDER = "saved_data"
RESULTS_FILE = "scalping_results.xlsx"
TICKER = "MARA"  # Replace with a valid ticker symbol
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-01-20"
INITIAL_BALANCE = 1000  # Starting balance
CHUNK_DAYS = 30  # Fetch intraday data in 30-day chunks
TIMEFRAME = "minute"  # Use 1-minute data for scalping

# Add new parameters for strategy optimization
PROFIT_TARGET_PCT = 0.002  # 0.2% profit target per trade
STOP_LOSS_PCT = 0.001  # 0.1% stop loss per trade
MAX_TRADES_PER_DAY = 10
MIN_VOLUME_THRESHOLD = 1000  # Minimum volume for trade entry

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
    # Optimize indicators for scalping
    data["EMA3"] = EMAIndicator(data["Close"], window=3).ema_indicator()  # Faster EMA
    data["EMA5"] = EMAIndicator(data["Close"], window=5).ema_indicator()
    data["EMA8"] = EMAIndicator(data["Close"], window=8).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=7).rsi()  # Shorter RSI window
    data["VWAP"] = VolumeWeightedAveragePrice(
        data["High"], data["Low"], data["Close"], data["Volume"]
    ).volume_weighted_average_price()

    # Add momentum and volatility features
    data["Price_Change"] = data["Close"].pct_change()
    data["Volume_Change"] = data["Volume"].pct_change()
    data["Volatility"] = data["Close"].rolling(window=5).std()

    return data.dropna()


# ======================
# TRAINING FUNCTION
# ======================
def train_model(data):
    if data.empty:
        print("Error: No training data available.")
        return None

    # Enhanced feature engineering
    features = [
        "EMA3",
        "EMA5",
        "EMA8",
        "RSI",
        "VWAP",
        "Price_Change",
        "Volume_Change",
        "Volatility",
    ]

    # Create more sophisticated target variable
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data["Return"] = data["Close"].pct_change().shift(-1)
    # Only consider significant moves as signals
    data.loc[abs(data["Return"]) < 0.001, "Target"] = 0

    data.dropna(inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(
        data[features], data["Target"], test_size=0.2, random_state=42
    )

    # Use more trees and better parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    return model


# ======================
# SCALPING STRATEGY
# ======================
def scalping_strategy(data, model, initial_balance=INITIAL_BALANCE):
    if data.empty or model is None:
        print("Error: No data or model available for scalping.")
        return pd.DataFrame(), pd.DataFrame()

    balance = initial_balance
    position = 0
    trade_logs = []
    daily_trades = 0
    last_trade_date = None
    entry_price = None

    # Generate prediction probabilities instead of just signals
    probabilities = model.predict_proba(
        data[
            [
                "EMA3",
                "EMA5",
                "EMA8",
                "RSI",
                "VWAP",
                "Price_Change",
                "Volume_Change",
                "Volatility",
            ]
        ]
    )
    data["Buy_Probability"] = probabilities[:, 1]

    # Pre-process dates to ensure we handle them consistently
    current_date = None

    for i in range(1, len(data)):
        current_time = data.index[i]

        # Check if we've moved to a new day
        if current_date != current_time.date():
            current_date = current_time.date()
            daily_trades = 0  # Reset counter at the start of each new day

        # Skip if we've hit the daily trade limit
        if daily_trades >= MAX_TRADES_PER_DAY:
            continue

        current_price = data["Close"].iloc[i]

        # Enhanced entry conditions
        if position == 0:
            if (
                data["Buy_Probability"].iloc[i] > 0.75  # Strong buy signal
                and data["Volume"].iloc[i] > MIN_VOLUME_THRESHOLD
                and data["EMA3"].iloc[i] > data["EMA8"].iloc[i]
                and data["RSI"].iloc[i] < 70
                and daily_trades < MAX_TRADES_PER_DAY
            ):

                risk_amount = balance * 0.01  # Risk 1% per trade
                position_size = risk_amount / (current_price * STOP_LOSS_PCT)
                shares_to_buy = min(int(position_size), int(balance // current_price))

                if shares_to_buy > 0:
                    position = shares_to_buy
                    entry_price = current_price
                    balance -= position * current_price
                    daily_trades += 1

                    trade_logs.append(
                        {
                            "Date": current_time,
                            "Action": "Buy",
                            "Price": current_price,
                            "Shares": position,
                            "Balance": balance,
                            "Signal_Strength": data["Buy_Probability"].iloc[i],
                        }
                    )

        # Enhanced exit conditions
        elif position > 0:
            profit_pct = (current_price - entry_price) / entry_price

            # Exit conditions
            should_exit = (
                profit_pct <= -STOP_LOSS_PCT  # Stop loss
                or profit_pct >= PROFIT_TARGET_PCT  # Take profit
                or (
                    data["Buy_Probability"].iloc[i] < 0.3 and profit_pct > 0
                )  # Exit on weak signal
            )

            if should_exit:
                balance += position * current_price
                exit_type = (
                    "Stop Loss"
                    if profit_pct <= -STOP_LOSS_PCT
                    else (
                        "Take Profit"
                        if profit_pct >= PROFIT_TARGET_PCT
                        else "Signal Exit"
                    )
                )

                trade_logs.append(
                    {
                        "Date": current_time,
                        "Action": exit_type,
                        "Price": current_price,
                        "Shares": position,
                        "Balance": balance,
                        "Profit_Pct": profit_pct * 100,
                    }
                )

                position = 0
                entry_price = None

    # Create detailed trade log
    trade_logs_df = pd.DataFrame(trade_logs)

    # Calculate enhanced summary metrics
    if not trade_logs_df.empty:
        profitable_trades = trade_logs_df[
            trade_logs_df["Action"].isin(["Take Profit", "Signal Exit"])
        ]
        total_trades = (
            len(trade_logs_df) // 2
        )  # Divide by 2 since each trade has entry and exit
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        avg_profit_pct = (
            profitable_trades["Profit_Pct"].mean() if not profitable_trades.empty else 0
        )
        max_drawdown = (data["Close"].cummax() - data["Close"]).max()

        summary = pd.DataFrame(
            {
                "Total Return": [(balance - initial_balance) / initial_balance],
                "Win Rate": [win_rate],
                "Average Profit %": [avg_profit_pct],
                "Max Drawdown": [max_drawdown],
                "Total Trades": [total_trades],
                "Profitable Trades": [len(profitable_trades)],
            }
        )
    else:
        summary = pd.DataFrame()

    return trade_logs_df, summary


# Execution
train_data = add_indicators(
    fetch_polygon_data(TICKER, TRAIN_START_DATE, TRAIN_END_DATE, timeframe=TIMEFRAME)
)
test_data = add_indicators(
    fetch_polygon_data(TICKER, TEST_START_DATE, TEST_END_DATE, timeframe=TIMEFRAME)
)

# Train the model
model = train_model(train_data)

# Run scalping strategy with the trained model
if model:
    trade_logs, summary = scalping_strategy(test_data, model)

    # Ensure we have data before writing to Excel
    if not trade_logs.empty or not summary.empty:
        # Create a new Excel writer object
        with pd.ExcelWriter(RESULTS_FILE, engine="openpyxl") as writer:
            # Write trade logs if we have any
            if not trade_logs.empty:
                trade_logs.to_excel(writer, sheet_name="Trade Logs", index=True)

            # Write summary if we have any
            if not summary.empty:
                summary.to_excel(writer, sheet_name="Summary", index=True)

        print(f"Results successfully saved to {RESULTS_FILE}")
    else:
        print("No trades were executed during the test period.")
else:
    print("Scalping strategy skipped due to model training failure.")
