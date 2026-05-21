#!/usr/bin/env python3
"""
Fetch daily OHLCV data for NSE symbols listed in nifty500.csv.
- Adds .NS suffix.
- Waits 0.7 seconds between requests.
- Skips symbols with no data.
- Saves each symbol as data/SYMBOL.json.
- Fetches 1 year of historical data.
"""

import os
import json
import time
import pandas as pd
import yfinance as yf

def load_symbols(csv_path="nifty500.csv"):
    df = pd.read_csv(csv_path)
    symbols = df['SYMBOL'].dropna().str.strip().tolist()
    # Remove any stray header value if present
    symbols = [s for s in symbols if s and s.upper() != 'SYMBOL']
    return symbols

def fetch_symbol(symbol, retries=2, delay=1):
    ticker = f"{symbol}.NS"
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            # CHANGE: 1 year of data instead of 3 months
            hist = stock.history(period="1y")
            if hist.empty:
                print(f"⚠️ No data for {ticker}")
                return None
            hist = hist.reset_index()
            hist = hist.rename(columns={
                'Date': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            hist['time'] = hist['time'].dt.strftime('%Y-%m-%d')
            hist = hist[['time', 'open', 'high', 'low', 'close', 'volume']]
            return hist.to_dict('records')
        except Exception as e:
            print(f"Error fetching {ticker} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def main():
    symbols = load_symbols()
    print(f"📊 Found {len(symbols)} symbols in nifty500.csv")

    os.makedirs("data", exist_ok=True)

    successful = 0
    failed = []

    for idx, sym in enumerate(symbols, 1):
        print(f"🔄 [{idx}/{len(symbols)}] Fetching {sym}.NS ...")
        data = fetch_symbol(sym)
        if data is not None:
            out_file = f"data/{sym}.json"
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved {sym}.NS - {len(data)} days")
            successful += 1
        else:
            print(f"⚠️ Skipping {sym}.NS: No data or error")
            failed.append(sym)

        # 0.7 second delay between requests
        time.sleep(0.7)

    print("\n=========================================")
    print(f"📊 FETCH SUMMARY:")
    print(f"   ✅ Successful: {successful} symbols")
    print(f"   ⚠️ Failed/Skipped: {len(failed)} symbols")
    print(f"   📈 Total processed: {len(symbols)} symbols")
    print("=========================================\n")

    if failed:
        print("⚠️ FAILED SYMBOLS:\n")
        for sym in failed:
            print(f"  - {sym}.NS")
        print("\n✅ Fetch completed with partial data")
    else:
        print("✅ All symbols fetched successfully!")

if __name__ == "__main__":
    main()
