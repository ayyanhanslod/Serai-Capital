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
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_{from_date}_{to_date}.csv")
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            return data
        except Exception as e:
            print(f"Error reading CSV file: {e}")

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
        final_data.to_csv(file_path)
        return final_data
    return pd.DataFrame()


# ======================
# TECHNICAL INDICATORS
# ======================
def add_indicators(data):
    if data.empty:
        print("Warning: Data is empty after fetching!")
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
        print("Error: No training data available.")
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
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing feature columns: {missing_cols}")
        return None

    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    data.dropna(inplace=True)
    if len(data) < 50:
        print(f"Error: Not enough data points for training ({len(data)} rows).")
        return None

    X_train, X_val, y_train, y_val = train_test_split(
        data[features], data["Target"], test_size=0.2, random_state=42
    )
    if X_train.empty:
        print("Error: Training set is empty after splitting.")
        return None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(
        f"Training Accuracy: {accuracy_score(y_val, model.predict(X_val)) * 100:.2f}%"
    )
    return model


# ======================
# BACKTESTING FUNCTION
# ======================
def backtest_strategy(data, model):
    if model is None or data.empty:
        print("Error: No model available for backtesting.")
        return pd.DataFrame(), pd.DataFrame()

    data["Signal"] = model.predict(
        data[
            [
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
        ]
    )
    return pd.DataFrame(), pd.DataFrame()


# Execution
train_data = add_indicators(
    fetch_polygon_data(TICKER, TRAIN_START_DATE, TRAIN_END_DATE)
)
model = train_model(train_data)

if model:
    test_data = add_indicators(
        fetch_polygon_data(TICKER, TEST_START_DATE, TEST_END_DATE)
    )
    trade_results, summary = backtest_strategy(test_data, model)
    with pd.ExcelWriter(RESULTS_FILE) as writer:
        trade_results.to_excel(writer, sheet_name="Trade Log", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"Results saved to {RESULTS_FILE}")
else:
    print("Backtesting skipped due to training failure.")
