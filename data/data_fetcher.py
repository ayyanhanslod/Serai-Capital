import pandas as pd
import requests
import matplotlib.pyplot as plt

# Polygon.io API Key and Ticker
API_KEY = "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"  # Replace with your API key
TICKER = "SOUN"


# Fetch Historical Data from Polygon.io
def fetch_polygon_data(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"apiKey": API_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            # Format the data
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


# Define Date Range
from_date = "2024-01-01"
to_date = "2024-12-31"

# Fetch Data
data = fetch_polygon_data(
    TICKER, multiplier=1, timespan="minute", from_date=from_date, to_date=to_date
)

print(data.head())

# Plot the Closing Prices
plt.figure(figsize=(14, 8))
plt.plot(data.index, data["Close"], label="Close Price", color="blue")
plt.title(f"Historical Close Prices for {TICKER} ({from_date} to {to_date})")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid()
plt.show()
