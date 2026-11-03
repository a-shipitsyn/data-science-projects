import argparse
import os

import ccxt
import pandas as pd

from backtester.data_quality import report_gaps

MAX_PER_REQUEST = 300   # Coinbase returns at most 300 candles per request


def download_coinbase_data(symbol: str = 'BTC/USD', timeframe: str = '1h', limit: int = 1000,
                           output: str | None = None) -> pd.DataFrame | None:
    """
    Downloads the last `limit` OHLCV candles from Coinbase, page by page, and saves them to a CSV.
    """
    print(f"Connecting to Coinbase to fetch {limit} candles for {symbol} ({timeframe})...")

    # Initialize the CCXT Coinbase exchange object (public data, no API keys needed)
    exchange = ccxt.coinbase({'enableRateLimit': True})
    step = exchange.parse_timeframe(timeframe) * 1000          # candle length in ms
    since = exchange.milliseconds() - limit * step

    try:
        # Fetch page by page: each returns a list of [timestamp, open, high, low, close, volume]
        rows = []
        while len(rows) < limit:
            page = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=MAX_PER_REQUEST)
            if not page:
                break
            rows.extend(page)
            since = page[-1][0] + step
            if since > exchange.milliseconds():
                break

        if not rows:
            print("No data returned from exchange.")
            return None

        # Convert to Pandas DataFrame, drop duplicates at page borders
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates('timestamp').sort_values('timestamp').tail(limit)

        # Convert milliseconds timestamp to readable datetime (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set timestamp as the index (crucial for time-series analysis)
        df = df.set_index('timestamp')

        # Format filename (e.g., data/historical/BTC-USD_1h.csv)
        filename = output or f"data/historical/{symbol.replace('/', '-')}_{timeframe}.csv"
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        # Save to CSV
        df.to_csv(filename)
        print(f"✅ Successfully saved {len(df)} rows to {filename}")
        report_gaps(df, filename)
        print(df.tail())  # Show the last 5 rows as a preview
        return df

    except (ccxt.BaseError, OSError) as e:
        print(f"❌ Error fetching data: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Download OHLCV candles from Coinbase")
    # Timeframes supported by Coinbase typically include: '1m', '5m', '15m', '1h', '6h', '1d'
    ap.add_argument("--symbol", default="BTC/USD", help="e.g. BTC/USD, ETH/USD, SOL/USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit", type=int, default=1000, help="number of most recent candles")
    ap.add_argument("--output", help="CSV path, default data/historical/<symbol>_<timeframe>.csv")
    args = ap.parse_args()
    download_coinbase_data(args.symbol, args.timeframe, args.limit, args.output)


if __name__ == "__main__":
    main()
