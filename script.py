#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import os
from pathlib import Path
import yfinance as yf

def fetch_batch_stocks(symbols_batch):
    """
    Fetch data for multiple stocks in ONE API call
    This sends 1 request for 20 stocks instead of 20 separate requests
    """
    try:
        # Download all symbols in this batch with ONE request
        data = yf.download(
            symbols=" ".join(symbols_batch),  # Join symbols with space
            period="3mo",
            group_by='ticker',
            threads=False,  # Disable threading to keep it as single request
            progress=False
        )
        
        # Handle single vs multiple tickers
        if len(symbols_batch) == 1:
            data = {symbols_batch[0]: data}
        
        return data
    except Exception as e:
        print(f"  ❌ Batch API call failed: {e}")
        return None

def save_stock_data(symbol, df):
    """Save individual stock data to JSON"""
    try:
        data = []
        for idx, row in df.iterrows():
            data.append({
                'time': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            })
        
        os.makedirs('data', exist_ok=True)
        with open(f'data/{symbol}.json', 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"    ❌ Error saving {symbol}: {e}")
        return False

def detect_patterns(df):
    """Your pattern detection logic here"""
    patterns = []
    # Your existing pattern detection code
    return patterns

def process_batch(batch_symbols, batch_num, total_batches):
    """
    Process ONE batch of stocks with a SINGLE API call
    """
    print(f"\n📦 BATCH {batch_num}/{total_batches}")
    print(f"   Stocks in this batch: {', '.join(batch_symbols)}")
    print(f"   Making 1 API call for {len(batch_symbols)} stocks...")
    print("-" * 50)
    
    # ONE API call for all stocks in this batch
    batch_data = fetch_batch_stocks(batch_symbols)
    
    if not batch_data:
        print(f"  ❌ Batch API call failed")
        return []
    
    batch_results = []
    
    # Process each stock from the batch response
    for symbol in batch_symbols:
        print(f"  📊 Processing {symbol}...")
        
        if symbol in batch_data:
            df = batch_data[symbol]
            if df is not None and not df.empty:
                # Save the data
                if save_stock_data(symbol, df):
                    # Detect patterns
                    patterns = detect_patterns(df)
                    batch_results.append({
                        'symbol': symbol,
                        'detected': patterns,
                        'data_points': len(df)
                    })
                    print(f"    ✅ Saved {len(df)} records")
                else:
                    print(f"    ❌ Failed to save")
            else:
                print(f"    ⚠️ No data returned")
        else:
            print(f"    ⚠️ Symbol not in response")
    
    return batch_results

def visualize_pattern(symbol, pattern_data):
    """Your existing visualization function"""
    try:
        with open(f'data/{symbol}.json', 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-60:], df['close'].values[-60:], 'b-', linewidth=1.5)
        
        pattern_text = "\n".join([f"{p['pattern']}: {p['confidence']}%" 
                                   for p in pattern_data['detected']])
        
        ax.text(0.02, 0.98, pattern_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_title(f"{symbol} - Chart Patterns Detected")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend(['Close Price'])
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        Path('shape.dna').mkdir(exist_ok=True)
        plt.savefig(f'shape.dna/{symbol}_pattern.png', dpi=100)
        plt.close()
        
        return True
    except Exception as e:
        print(f"  ❌ Visualization error for {symbol}: {e}")
        return False

def main():
    """Main function - processes 500 stocks in batches of 20 (25 API calls total)"""
    
    # Load stock symbols from nifty500.csv
    print("\n" + "="*60)
    print("📈 NIFTY 500 BATCH PROCESSING")
    print("="*60)
    
    try:
        df = pd.read_csv('nifty500.csv')
        # Try to find the symbol column
        symbol_col = None
        for col in ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock']:
            if col in df.columns:
                symbol_col = col
                break
        
        if symbol_col:
            all_stocks = df[symbol_col].dropna().tolist()
        else:
            # Assume first column
            all_stocks = df.iloc[:, 0].dropna().tolist()
        
        # Clean symbols
        all_stocks = [str(s).strip().upper() for s in all_stocks]
        print(f"✅ Loaded {len(all_stocks)} stocks from nifty500.csv")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # BATCH CONFIGURATION
    BATCH_SIZE = 20  # 20 stocks per API call
    DELAY_BETWEEN_BATCHES = 2  # 2 seconds between batches (adjust as needed)
    
    total_batches = (len(all_stocks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Batch size: {BATCH_SIZE} stocks per API call")
    print(f"   • Total batches: {total_batches} API calls")
    print(f"   • Delay between batches: {DELAY_BETWEEN_BATCHES} seconds")
    print(f"   • Total API calls for {len(all_stocks)} stocks: {total_batches}")
    
    confirm = input(f"\n🚀 Start processing? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled")
        return
    
    all_results = []
    
    # Process batches - EACH BATCH = 1 API CALL for 20 stocks
    for i in range(0, len(all_stocks), BATCH_SIZE):
        batch = all_stocks[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        # ONE API call for this entire batch of 20 stocks
        batch_results = process_batch(batch, batch_num, total_batches)
        all_results.extend(batch_results)
        
        # Show progress
        progress = (batch_num / total_batches) * 100
        print(f"\n📈 Progress: {batch_num}/{total_batches} batches ({progress:.1f}%)")
        print(f"   ✅ Successful: {len(batch_results)}/{len(batch)} stocks in this batch")
        print(f"   📊 Total so far: {len(all_results)}/{len(all_stocks)} stocks")
        
        # Save intermediate results
        os.makedirs('shape.dna', exist_ok=True)
        with open('shape.dna/patterns.json', 'w') as f:
            json.dump({'detailed_results': all_results}, f, indent=2)
        
        # Delay between batches (except after last)
        if i + BATCH_SIZE < len(all_stocks):
            print(f"\n⏳ Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch API call...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    print("\n" + "="*60)
    print(f"✅ DATA FETCHING COMPLETE!")
    print(f"   • Total API calls made: {total_batches}")
    print(f"   • Stocks processed: {len(all_results)}/{len(all_stocks)}")
    print(f"   • Data saved in 'data/' directory")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*60)
    
    for idx, stock_data in enumerate(all_results, 1):
        print(f"  [{idx}/{len(all_results)}] Visualizing {stock_data['symbol']}...")
        visualize_pattern(stock_data['symbol'], stock_data)
    
    print(f"\n✅ Complete! Check 'shape.dna/' folder for visualizations")

if __name__ == "__main__":
    main()
