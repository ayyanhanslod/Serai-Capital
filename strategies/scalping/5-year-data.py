import os
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import requests
from datetime import datetime, timedelta
import xlsxwriter

# Parameters
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon.io API key
DATA_FOLDER = "saved_data"  # Folder to save fetched data
TICKERS = ["NVDA"]  # List of stocks to backtest
TRANSACTION_COST = 0.0035  # Commission per share
STOP_LOSS_PERCENT = 3 / 100  # Stop-loss threshold as 3%
INITIAL_BALANCE = 1000  # Starting balance in USD
CHUNK_DAYS = 30  # Number of days per chunk when fetching data

# Training and Testing date ranges
TRAIN_START_DATE = "2024-01-01"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-01-10"

# Ensure the data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)


# Fetch data in chunks
def fetch_polygon_data_chunked(ticker, from_date, to_date):
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_{from_date}_{to_date}.csv")
    if os.path.exists(file_path):
        print(f"Loading data for {ticker} from {file_path}...")
        return pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")

    print(f"Fetching data for {ticker} from {from_date} to {to_date}...")
    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    all_data = []

    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunk_from = start_date.strftime("%Y-%m-%d")
        chunk_to = chunk_end_date.strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{chunk_from}/{chunk_to}"
        params = {"apiKey": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "results" in data:
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
            else:
                print(f"No data found for {ticker} from {chunk_from} to {chunk_to}.")
        else:
            raise ValueError(
                f"Error fetching data: {response.status_code}, {response.text}"
            )

        start_date = chunk_end_date + timedelta(days=1)

    if all_data:
        final_data = pd.concat(all_data)
        final_data.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}.")
        return final_data
    else:
        raise ValueError(f"No data retrieved for {ticker} in the given date range.")


# Add technical indicators
def add_indicators(data):
    data["EMA9"] = EMAIndicator(data["Close"], window=9).ema_indicator()
    data["EMA21"] = EMAIndicator(data["Close"], window=21).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()

    data["Price_Change"] = data["Close"].pct_change() * 100
    data["Volatility"] = (data["High"] - data["Low"]) / data["Close"] * 100
    return data.dropna()


# Train a machine learning model
def train_model(data):
    data["Target"] = np.where(data["Close"].shift(-5) > data["Close"], 1, 0)

    features = [
        "EMA9",
        "EMA21",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "Price_Change",
        "Volatility",
    ]
    X = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {grid_search.score(X_test, y_test):.4f}")

    return grid_search.best_estimator_


# Backtest the strategy using predictions
def backtest_strategy(ticker, data, stop_loss_percent, model):
    features = [
        "EMA9",
        "EMA21",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "Price_Change",
        "Volatility",
    ]
    X = data[features]
    data["Prediction"] = model.predict(X)

    balance = INITIAL_BALANCE
    shares_held = 0
    entry_price = None
    trade_log = []

    for index, row in data.iterrows():
        price = row["Close"]
        if price <= 0:
            continue

        if shares_held > 0 and price <= entry_price * (1 - stop_loss_percent):
            revenue = shares_held * price
            commission = shares_held * TRANSACTION_COST
            balance += revenue - commission
            trade_log.append((index, "STOP-LOSS SELL", shares_held, price, balance))
            shares_held = 0
            entry_price = None

        elif row["Prediction"] == 1 and balance > 0:
            shares_to_buy = int(balance // (price + TRANSACTION_COST))
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                commission = shares_to_buy * TRANSACTION_COST
                balance -= cost + commission
                shares_held += shares_to_buy
                entry_price = price
                trade_log.append((index, "BUY", shares_to_buy, price, balance))

        elif row["Prediction"] == 0 and shares_held > 0:
            revenue = shares_held * price
            commission = shares_held * TRANSACTION_COST
            balance += revenue - commission
            trade_log.append((index, "SELL", shares_held, price, balance))
            shares_held = 0
            entry_price = None

    final_balance = balance + (shares_held * data.iloc[-1]["Close"])
    final_balance -= shares_held * TRANSACTION_COST
    net_profit = final_balance - INITIAL_BALANCE
    win_rate = (
        sum([1 for t in trade_log if "SELL" in t[1] and t[2] > 0]) / len(trade_log)
        if trade_log
        else 0
    )

    trade_log_df = pd.DataFrame(
        trade_log, columns=["Date", "Action", "Shares", "Price", "Balance"]
    )

    return {
        "Ticker": ticker,
        "Net Profit": net_profit,
        "Final Balance": final_balance,
        "Win Rate": win_rate,
        "Total Trades": len(trade_log),
        "Trade Log": trade_log_df,
    }


# Run the backtest for multiple stocks and save results
results = []
excel_file = "backtest_results_with_logs.xlsx"
writer = pd.ExcelWriter(excel_file, engine="xlsxwriter")

for ticker in TICKERS:
    try:
        print(f"Processing {ticker}...")

        train_data = fetch_polygon_data_chunked(
            ticker,
            from_date=TRAIN_START_DATE,
            to_date=TRAIN_END_DATE,
        )
        train_data = add_indicators(train_data)

        model = train_model(train_data)

        test_data = fetch_polygon_data_chunked(
            ticker,
            from_date=TEST_START_DATE,
            to_date=TEST_END_DATE,
        )
        test_data = add_indicators(test_data)

        result = backtest_strategy(ticker, test_data, STOP_LOSS_PERCENT, model)
        results.append(result)

        result["Trade Log"].to_excel(writer, sheet_name=f"{ticker}_Log", index=False)

    except Exception as e:
        print(f"Error with {ticker}: {e}")

summary_df = pd.DataFrame(
    [
        {
            "Ticker": r["Ticker"],
            "Net Profit": r["Net Profit"],
            "Final Balance": r["Final Balance"],
            "Win Rate": r["Win Rate"],
            "Total Trades": r["Total Trades"],
        }
        for r in results
    ]
)
summary_df.to_excel(writer, sheet_name="Summary", index=False)

writer.close()
print(f"Backtesting results saved to {excel_file}.")
