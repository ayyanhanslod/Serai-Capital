import os
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
import requests

# Parameters
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon.io API key
DATA_FOLDER = "saved_data"  # Folder to save/load data
RESULTS_FILE = "swing_trading_results.xlsx"  # Output Excel file for results
TICKER = "MARA"  # Example stock for swing trading
INITIAL_BALANCE = 1000  # Starting balance
STOP_LOSS_PERCENT = 0.02  # Tighter stop loss at 2%
TAKE_PROFIT_PERCENT = 0.05  # More realistic take profit at 5%
RSI_OVERBOUGHT = 75  # Slightly higher for stronger confirmation
RSI_OVERSOLD = 25  # Slightly lower for stronger confirmation
CHUNK_DAYS = 30  # Number of days per fetch chunk

# Specify training and testing date ranges
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2025-01-24"

# Ensure the data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)


def fetch_polygon_data(ticker, from_date, to_date):
    """Fetch data from Polygon API"""
    print(f"Fetching data for {ticker} from {from_date} to {to_date}...")
    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d")
    all_data = []

    while start_date < end_date:
        chunk_end_date = min(start_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        chunk_from = start_date.strftime("%Y-%m-%d")
        chunk_to = chunk_end_date.strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{chunk_from}/{chunk_to}"
        params = {"apiKey": API_KEY}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df = pd.DataFrame(data["results"])
                df["Date"] = pd.to_datetime(df["t"], unit="ms")
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
                all_data.append(df[["Date", "Open", "High", "Low", "Close", "Volume"]])
            else:
                print(f"No data found for {ticker} from {chunk_from} to {chunk_to}.")
        else:
            print(f"Error fetching data: {response.status_code}, {response.text}")

        start_date = chunk_end_date + timedelta(days=1)

    if all_data:
        final_data = pd.concat(all_data)
        final_data.set_index("Date", inplace=True)
        return final_data
    else:
        raise ValueError(f"No data retrieved for {ticker} in the given date range.")


def load_or_fetch_data(ticker, from_date, to_date):
    """Load saved data or fetch new data if missing"""
    file_path = os.path.join(DATA_FOLDER, f"{ticker}_daily_data.xlsx")

    if os.path.exists(file_path):
        print(f"Loading saved data for {ticker} from {file_path}...")
        data = pd.read_excel(file_path, parse_dates=["Date"], index_col="Date")
        return data[(data.index >= from_date) & (data.index <= to_date)]
    else:
        print(f"No saved data found for {ticker}. Fetching new data...")
        data = fetch_polygon_data(ticker, from_date, to_date)
        data.to_excel(file_path)
        print(f"Data for {ticker} saved to {file_path}.")
        return data


def add_indicators(data):
    """Add enhanced technical indicators"""
    # Keep existing indicators
    data["SMA50"] = SMAIndicator(data["Close"], window=50).sma_indicator()
    data["SMA200"] = SMAIndicator(data["Close"], window=200).sma_indicator()
    data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

    # Add volume-weighted average price (VWAP)
    data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()

    # Add Average True Range for dynamic stop loss
    data["TR"] = np.maximum(
        data["High"] - data["Low"],
        np.maximum(
            abs(data["High"] - data["Close"].shift(1)),
            abs(data["Low"] - data["Close"].shift(1)),
        ),
    )
    data["ATR"] = data["TR"].rolling(window=14).mean()

    return data.dropna()


def backtest_strategy(data):
    """Enhanced backtest strategy"""
    balance = INITIAL_BALANCE
    shares_held = 0
    entry_price = None
    trade_log = []
    consecutive_losses = 0
    max_consecutive_losses = 3  # Risk management

    for index, row in data.iterrows():
        price = row["Close"]

        # Enhanced trend confirmation
        is_trending_up = (row["SMA50"] > row["SMA200"]) and (price > row["VWAP"])
        is_trending_down = (row["SMA50"] < row["SMA200"]) and (price < row["VWAP"])

        # Position sizing based on ATR
        position_size = min(0.02 * balance, balance)  # Risk 2% per trade

        if shares_held == 0 and consecutive_losses < max_consecutive_losses:
            # Enhanced entry conditions
            if (
                is_trending_up
                and row["RSI"] < RSI_OVERSOLD
                and row["Volume"] > data["Volume"].rolling(20).mean()
            ):  # Volume confirmation

                shares_to_buy = int(position_size // price)
                if shares_to_buy > 0:
                    balance -= shares_to_buy * price
                    shares_held += shares_to_buy
                    entry_price = price
                    trade_log.append((index, "BUY", shares_to_buy, price, balance))

        if shares_held > 0:
            # Dynamic stop loss based on ATR
            stop_loss_price = max(
                entry_price * (1 - STOP_LOSS_PERCENT),
                entry_price - (2 * row["ATR"]),  # 2 ATR stop loss
            )
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)

            if price <= stop_loss_price:
                # Stop loss hit
                balance += shares_held * price
                trade_log.append((index, "STOP LOSS", shares_held, price, balance))
                shares_held = 0
                entry_price = None
                consecutive_losses += 1

            elif price >= take_profit_price:
                # Take profit hit
                balance += shares_held * price
                trade_log.append((index, "TAKE PROFIT", shares_held, price, balance))
                shares_held = 0
                entry_price = None
                consecutive_losses = 0  # Reset consecutive losses on winning trade

            elif (
                is_trending_down
                and row["RSI"] > RSI_OVERBOUGHT
                and row["Volume"] > data["Volume"].rolling(20).mean()
            ):
                # Enhanced exit conditions with volume confirmation
                balance += shares_held * price
                trade_log.append((index, "SELL", shares_held, price, balance))
                shares_held = 0
                entry_price = None

    final_balance = balance + (
        shares_held * data.iloc[-1]["Close"] if shares_held > 0 else 0
    )
    net_profit = final_balance - INITIAL_BALANCE
    trade_log_df = pd.DataFrame(
        trade_log, columns=["Date", "Action", "Shares", "Price", "Balance"]
    )
    return {
        "Final Balance": final_balance,
        "Net Profit": net_profit,
        "Total Trades": len(trade_log),
        "Trade Log": trade_log_df,
    }


def main():
    test_data = load_or_fetch_data(TICKER, TEST_START_DATE, TEST_END_DATE)
    test_data = add_indicators(test_data)
    result = backtest_strategy(test_data)

    with pd.ExcelWriter(RESULTS_FILE) as writer:
        result["Trade Log"].to_excel(writer, sheet_name="Trade Log", index=False)
        summary_df = pd.DataFrame(
            {
                "Ticker": [TICKER],
                "Final Balance": [result["Final Balance"]],
                "Net Profit": [result["Net Profit"]],
                "Total Trades": [result["Total Trades"]],
            }
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Results saved to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    main()
