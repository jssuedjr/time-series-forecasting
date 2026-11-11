from pathlib import Path
import yfinance as yf

# 1. Get the absolute directory of the current script
script_dir = Path(__file__).resolve().parent
csv_path = script_dir / "tsla_historical_data.csv"

# 2. Combine the directory with your desired filename
csv_path = script_dir / "tsla_historical_data.csv"

## new from gpt
# 3. Save the DataFrame
df = yf.download(
    "TSLA",
    period="5y",
    interval="1d",
    auto_adjust=False,
    multi_level_index=False
)

## old from gpt 
#- Adjusted Close: A historical price series adjusted for stock splits and dividends.
ticker = yf.Ticker("TSLA")
df = ticker.history(period="5y")
df.to_csv(csv_path)