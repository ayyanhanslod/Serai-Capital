import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import requests

# Define API Key and Parameters
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your Polygon.io API key
TICKER = "MARA"
INITIAL_BALANCE = 1000  # Starting balance in USD
STOP_LOSS_PERCENT = 0.5 / 100  # Stop-loss threshold as 0.5%
print("Starting Balance: " + str(INITIAL_BALANCE))


# Step 1: Fetch Historical Data
def fetch_polygon_data(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"apiKey": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            # Format the data
            df["timestamp"] = pd.to_datetime(
                df["t"], unit="ms"
            )  # Convert UNIX time to datetime
            df.set_index("timestamp", inplace=True)  # Set timestamp as the index
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


from_date = "2024-10-01"
to_date = "2024-12-31"
data = fetch_polygon_data(
    TICKER, multiplier=1, timespan="minute", from_date=from_date, to_date=to_date
)
data.dropna(inplace=True)

# Step 2: Add Technical Indicators
data["EMA9"] = EMAIndicator(data["Close"], window=9).ema_indicator()
data["EMA21"] = EMAIndicator(data["Close"], window=21).ema_indicator()
data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

# Add MACD Indicator
macd = MACD(data["Close"])
data["MACD"] = macd.macd()
data["MACD_Signal"] = macd.macd_signal()
data["MACD_Hist"] = macd.macd_diff()

data.dropna(inplace=True)

# Step 3: Prepare Features for Machine Learning
data["Target"] = np.where(data["Close"].shift(-5) > data["Close"], 1, 0)

features = ["EMA9", "EMA21", "RSI", "MACD", "MACD_Signal", "MACD_Hist"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Simulate Paper Trading with Stop-Loss
balance = INITIAL_BALANCE
shares_held = 0
entry_price = None  # Track entry price for stop-loss
trade_log = []
transactions_count = 0

data["Prediction"] = model.predict(X)

for index, row in data.iterrows():
    price = row["Close"]

    # Stop-Loss Condition
    if shares_held > 0 and price <= entry_price * (1 - STOP_LOSS_PERCENT):
        revenue = shares_held * price
        balance += revenue
        transactions_count += 1
        trade_log.append((index, "STOP-LOSS SELL", shares_held, price, balance))
        shares_held = 0
        entry_price = None  # Reset entry price

    # Buy Signal
    elif row["Prediction"] == 1 and balance > 0:
        shares_to_buy = balance // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            balance -= cost
            shares_held += shares_to_buy
            entry_price = price  # Set entry price for stop-loss tracking
            transactions_count += 1
            trade_log.append((index, "BUY", shares_to_buy, price, balance))

    # Sell Signal
    elif row["Prediction"] == 0 and shares_held > 0:
        revenue = shares_held * price
        balance += revenue
        transactions_count += 1
        trade_log.append((index, "SELL", shares_held, price, balance))
        shares_held = 0
        entry_price = None  # Reset entry price

# Step 6: Print Trade Log and Results
for trade in trade_log:
    print(
        f"{trade[0]}: {trade[1]} {trade[2]} shares at ${trade[3]:.2f}, Balance: ${trade[4]:.2f}"
    )

final_balance = balance + (shares_held * data.iloc[-1]["Close"])
print(f"Final Balance: ${final_balance:.2f}")
print(f"Net Profit: ${final_balance - INITIAL_BALANCE:.2f}")
print(f"Total Transactions: {transactions_count}")

# Step 7: Plot Results
plt.figure(figsize=(14, 8))
plt.plot(data.index, data["Close"], label="Close Price", color="blue")
plt.plot(data.index, data["EMA9"], label="EMA9", color="green", linestyle="--")
plt.plot(data.index, data["EMA21"], label="EMA21", color="red", linestyle="--")

# Plot Buy and Sell Signals
buy_signals = data[data["Prediction"] == 1]
sell_signals = data[data["Prediction"] == 0]
plt.scatter(
    buy_signals.index,
    buy_signals["Close"],
    label="Buy Signal",
    marker="^",
    color="green",
    alpha=1,
)
plt.scatter(
    sell_signals.index,
    sell_signals["Close"],
    label="Sell Signal",
    marker="v",
    color="red",
    alpha=1,
)

plt.title(f"Scalping Strategy with ML, MACD, and Stop-Loss for {TICKER}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
