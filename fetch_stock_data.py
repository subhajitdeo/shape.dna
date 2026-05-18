#!/usr/bin/env python3
"""
Fetches daily OHLCV data for symbols from nifty500.csv.
- Adds .NS suffix for NSE.
- Waits 0.7 seconds between requests to avoid rate limiting.
- Skips symbols that return no data or raise errors.
- Saves each symbol as data/SYMBOL.json (format compatible with detect_patterns.py).
"""

import os
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime

def load_symbols(csv_path="nifty500.csv"):
    """Read symbols from CSV, skip header, clean up."""
    df = pd.read_csv(csv_path)
    # Assuming column name is 'SYMBOL' (as in your file)
    symbols = df['SYMBOL'].dropna().str.strip().tolist()
    # Remove any obvious non-symbols like 'SYMBOL' header if present
    symbols = [s for s in symbols if s and s.upper() != 'SYMBOL']
    return symbols

def fetch_symbol(symbol, retries=2, delay=1):
    """Fetch data for one symbol using yfinance. Return DataFrame or None."""
    ticker = f"{symbol}.NS"
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            # Get last 90 days (or more) – we need at least 60 for pattern detection
            hist = stock.history(period="3mo")
            if hist.empty:
                print(f"⚠️ No data for {ticker}")
                return None
            # Reset index to have 'time' column
            hist = hist.reset_index()
            # Rename columns to match detect_patterns.py expectations
            hist = hist.rename(columns={
                'Date': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            # Convert time to string ISO format (detect_patterns.py uses pd.to_datetime)
            hist['time'] = hist['time'].dt.strftime('%Y-%m-%d')
            # Keep only needed columns
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
        
        # Wait 0.7 seconds between requests to avoid rate limiting
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
