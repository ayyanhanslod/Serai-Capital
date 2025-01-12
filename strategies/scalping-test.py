import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import requests

# Parameters
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon.io API key
TICKERS = ["MARA", "PLTR", "TSLA", "NVDA", "SOUN"]  # List of stocks to backtest
TRANSACTION_COST = 0.0035  # Commission per share
STOP_LOSS_PERCENT = 3 / 100  # Stop-loss threshold as 3%
INITIAL_BALANCE = 1000  # Starting balance in USD
RESULTS_FILE = "backtest_results.xlsx"  # Output Excel file for results


# Function to fetch data from Polygon.io
def fetch_polygon_data(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
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
            return df[["Open", "High", "Low", "Close", "Volume"]]
        else:
            raise ValueError("No data found in the response.")
    else:
        raise ValueError(
            f"Error fetching data: {response.status_code}, {response.text}"
        )


# Function to read or fetch data
def get_data(ticker, from_date, to_date):
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)

    # Fetch the data directly from Polygon if it is not available in saved files
    print(f"Fetching data for {ticker} from Polygon.io...")
    data = fetch_polygon_data(
        ticker,
        multiplier=1,
        timespan="minute",
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
    )

    return data


# Add technical indicators
def add_indicators(data):
    data["EMA5"] = EMAIndicator(data["Close"], window=9).ema_indicator()
    data["EMA15"] = EMAIndicator(data["Close"], window=21).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

    macd = MACD(data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()
    return data.dropna()


# Backtest the strategy
def backtest_strategy(ticker, data, stop_loss_percent):
    # Prepare features and target
    data["Target"] = np.where(data["Close"].shift(-5) > data["Close"], 1, 0)
    features = ["EMA5", "EMA15", "RSI", "MACD", "MACD_Signal", "MACD_Hist"]
    X = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Simulate trading
    balance = INITIAL_BALANCE
    shares_held = 0
    entry_price = None
    trade_log = []
    data["Prediction"] = model.predict(X)

    for index, row in data.iterrows():
        price = row["Close"]

        # Ensure price is valid (non-zero)
        if price <= 0:
            continue

        # Stop-Loss Condition
        if shares_held > 0 and price <= entry_price * (1 - stop_loss_percent):
            revenue = shares_held * price
            commission = shares_held * TRANSACTION_COST
            balance += revenue - commission
            trade_log.append((index, "STOP-LOSS SELL", shares_held, price, balance))
            shares_held = 0
            entry_price = None

        # Buy Signal
        elif row["Prediction"] == 1 and balance > 0:
            shares_to_buy = int(balance // (price + TRANSACTION_COST))
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                commission = shares_to_buy * TRANSACTION_COST
                balance -= cost + commission
                shares_held += shares_to_buy
                entry_price = price
                trade_log.append((index, "BUY", shares_to_buy, price, balance))

        # Sell Signal
        elif row["Prediction"] == 0 and shares_held > 0:
            revenue = shares_held * price
            commission = shares_held * TRANSACTION_COST
            balance += revenue - commission
            trade_log.append((index, "SELL", shares_held, price, balance))
            shares_held = 0
            entry_price = None

    final_balance = balance + (shares_held * data.iloc[-1]["Close"])
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
writer = pd.ExcelWriter(RESULTS_FILE, engine="xlsxwriter")

for ticker in TICKERS:
    try:
        print(f"Processing {ticker}...")
        data = get_data(
            ticker,
            from_date="2024-12-30",
            to_date="2024-12-31",
        )
        data = add_indicators(data)
        result = backtest_strategy(ticker, data, STOP_LOSS_PERCENT)
        results.append(result)

        # Save trade log to a separate sheet
        result["Trade Log"].to_excel(writer, sheet_name=f"{ticker}_Log", index=False)

    except Exception as e:
        print(f"Error with {ticker}: {e}")

# Save summary results
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
print(f"Backtesting results saved to {RESULTS_FILE}.")
