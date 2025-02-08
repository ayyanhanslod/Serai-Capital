import os
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.utils import dropna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timedelta

# ======================
# PARAMETERS & SETTINGS
# ======================
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"
DATA_FOLDER = "saved_data"
RESULTS_FILE = "backtest_results.xlsx"
TICKER = "PLTR"
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-01-10"
INITIAL_BALANCE = 1000
CHUNK_DAYS = 30

os.makedirs(DATA_FOLDER, exist_ok=True)


# ======================
# DATA FETCHING FUNCTION
# ======================
def fetch_polygon_data(ticker, from_date, to_date):
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_{from_date}_{to_date}.xlsx")
    if os.path.exists(file_path):
        return pd.read_excel(file_path, parse_dates=["Date"], index_col="Date")

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    all_data = []

    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{chunk_end_date.strftime('%Y-%m-%d')}?apiKey={API_KEY}"
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

        start_date = chunk_end_date + timedelta(days=1)

    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_excel(file_path)
        return final_data
    return pd.DataFrame()


# ======================
# TECHNICAL INDICATORS
# ======================
def add_indicators(data):
    if data.empty:
        return data

    data = dropna(data)
    data["EMA9"] = EMAIndicator(data["Close"], window=9).ema_indicator()
    data["EMA21"] = EMAIndicator(data["Close"], window=21).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()
    data["Stoch_K"] = StochasticOscillator(
        data["High"], data["Low"], data["Close"]
    ).stoch()
    data["Stoch_D"] = StochasticOscillator(
        data["High"], data["Low"], data["Close"]
    ).stoch_signal()
    bb = BollingerBands(data["Close"])
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()
    data["OBV"] = OnBalanceVolumeIndicator(
        data["Close"], data["Volume"]
    ).on_balance_volume()
    return data.dropna()


# ======================
# TRAINING FUNCTION
# ======================
def train_model(data):
    if data.empty:
        print("No training data available.")
        return None

    features = [
        "EMA9",
        "EMA21",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Stoch_K",
        "Stoch_D",
        "BB_High",
        "BB_Low",
        "OBV",
    ]
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    X_train, X_val, y_train, y_val = train_test_split(
        data[features], data["Target"], test_size=0.2, random_state=42
    )

    if X_train.empty:
        print("Not enough data to train the model.")
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"Training Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    return model


# ======================
# BACKTESTING FUNCTION
# ======================
def backtest_strategy(data, model):
    if model is None or data.empty:
        print("No model available for backtesting.")
        return pd.DataFrame(), pd.DataFrame()

    features = [
        "EMA9",
        "EMA21",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Stoch_K",
        "Stoch_D",
        "BB_High",
        "BB_Low",
        "OBV",
    ]
    data["Signal"] = model.predict(data[features])
    balance = INITIAL_BALANCE
    shares_held = 0
    trade_log = []

    for index, row in data.iterrows():
        price = row["Close"]
        if row["Signal"] == 1 and shares_held == 0:
            shares_held = balance // price
            balance -= shares_held * price
            trade_log.append((index, "BUY", shares_held, price, balance))
        elif row["Signal"] == 0 and shares_held > 0:
            balance += shares_held * price
            trade_log.append((index, "SELL", shares_held, price, balance))
            shares_held = 0

    final_balance = balance + (
        shares_held * data.iloc[-1]["Close"] if shares_held > 0 else 0
    )
    summary = pd.DataFrame({"Final Balance": [final_balance]})
    return (
        pd.DataFrame(
            trade_log, columns=["Date", "Action", "Shares", "Price", "Balance"]
        ),
        summary,
    )


# Execution
train_data = fetch_polygon_data(TICKER, TRAIN_START_DATE, TRAIN_END_DATE)
train_data = add_indicators(train_data)
model = train_model(train_data)
test_data = fetch_polygon_data(TICKER, TEST_START_DATE, TEST_END_DATE)
test_data = add_indicators(test_data)
trade_results, summary = backtest_strategy(test_data, model)

with pd.ExcelWriter(RESULTS_FILE) as writer:
    trade_results.to_excel(writer, sheet_name="Trade Log", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
print(f"Results saved to {RESULTS_FILE}")
