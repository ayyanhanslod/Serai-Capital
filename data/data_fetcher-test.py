import requests
import pandas as pd


# Function to fetch data from Polygon API
def fetch_polygon_data(ticker, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    params = {"apiKey": "CTHBeNv4eb4B9eGOaDjXifsbeTV4kU4B"}

    # Fetch data from the API
    response = requests.get(url, params=params)
    data = response.json()

    if "results" in data:
        df = pd.DataFrame(data["results"])

        # Convert timestamp to readable format
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Rename the columns for better understanding
        df.rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"},
            inplace=True,
        )

        # Save the DataFrame to an Excel file
        df.to_excel(f"{ticker}_{from_date}_{to_date}_data.xlsx", index=True)

        print(
            f"Data for {ticker} from {from_date} to {to_date} has been saved to Excel."
        )
        return df
    else:
        print("No data found.")
        return pd.DataFrame()


# Example usage
from_date = "2024-01-01"
to_date = "2024-12-31"
ticker = "TSLA"
data = fetch_polygon_data(ticker, from_date, to_date)
