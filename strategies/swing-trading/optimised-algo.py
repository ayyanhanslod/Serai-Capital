import os
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.utils import dropna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timedelta
import xlsxwriter

# ======================
# PARAMETERS & SETTINGS
# ======================
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon.io API key
DATA_FOLDER = "saved_data"  # Folder to save fetched data
TICKER = "PLTR"  # Stock ticker symbol
TRAIN_START_DATE = "2024-01-01"  # Training start date
TRAIN_END_DATE = "2024-12-31"  # Training end date
TEST_START_DATE = "2025-01-01"  # Test start date
TEST_END_DATE = "2025-01-10"  # Test end date
INITIAL_BALANCE = 1000  # Starting capital
CHUNK_DAYS = 30  # Number of days per data chunk

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)


# ======================
# DATA FETCHING FUNCTION
# ======================
def fetch_polygon_data(ticker, from_date, to_date):
    """Fetch historical stock data from Polygon.io API."""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}{from_date}{to_date}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    all_data = []

    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date.strftime('%Y-%m-%d')}/{chunk_end_date.strftime('%Y-%m-%d')}?apiKey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                df.set_index("timestamp", inplace=True)
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
    """Add technical indicators to the dataset."""
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
# SAVE RESULTS TO EXCEL
# ======================
def save_results_to_excel(trade_results, summary):
    """Save backtest trade logs and summary results to an Excel file."""
    excel_file = "backtest_results.xlsx"
    writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")

    trade_results.to_excel(writer, sheet_name="Trade Log", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

    writer.close()
    print(f"Results saved to {excel_file}")


# ======================
# EXECUTION
# ======================
train_data = fetch_polygon_data(TICKER, TRAIN_START_DATE, TRAIN_END_DATE)
train_data = add_indicators(train_data)
optimized_model = train_optimized_model(train_data)
test_data = fetch_polygon_data(TICKER, TEST_START_DATE, TEST_END_DATE)
trade_results = backtest_optimized_strategy(test_data, optimized_model)
summary = pd.DataFrame(
    {"Final Balance": [INITIAL_BALANCE]}
)  # Add relevant summary calculations
save_results_to_excel(
    pd.DataFrame(
        trade_results, columns=["Date", "Action", "Shares", "Price", "Balance"]
    ),
    summary,
)
