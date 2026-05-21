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
    Fetch data for 50 stocks in ONE API call
    """
    try:
        # ONE API call for 50 stocks
        data = yf.download(
            symbols=" ".join(symbols_batch),
            period="3mo",
            group_by='ticker',
            threads=False,
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
    # Add your existing pattern detection code
    return patterns

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
        print(f"  ❌ Visualization error: {e}")
        return False

def main():
    """Process 500 stocks in batches of 50 per API call"""
    
    print("\n" + "="*60)
    print("📈 NIFTY 500 BATCH PROCESSING")
    print("="*60)
    
    # Load all 500 stocks from CSV
    try:
        df = pd.read_csv('nifty500.csv')
        
        # Find symbol column
        symbol_col = None
        for col in ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock']:
            if col in df.columns:
                symbol_col = col
                break
        
        if symbol_col:
            all_stocks = df[symbol_col].dropna().tolist()
        else:
            all_stocks = df.iloc[:, 0].dropna().tolist()
        
        # Clean symbols
        all_stocks = [str(s).strip().upper() for s in all_stocks]
        print(f"✅ Loaded {len(all_stocks)} stocks from nifty500.csv")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # BATCH CONFIGURATION - 50 stocks per API call
    BATCH_SIZE = 50  # 50 stocks per API request
    DELAY_BETWEEN_BATCHES = 2  # 2 seconds between batches
    
    total_stocks = len(all_stocks)
    total_batches = (total_stocks + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Batch size: {BATCH_SIZE} stocks per API call")
    print(f"   • Total stocks: {total_stocks}")
    print(f"   • Total API calls needed: {total_batches}")
    print(f"   • Delay between API calls: {DELAY_BETWEEN_BATCHES} seconds")
    
    if total_batches == 10:
        print(f"   • For 500 stocks: {total_batches} API calls (50 stocks × 10 calls = 500)")
    
    confirm = input(f"\n🚀 Start processing {total_batches} API calls? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled")
        return
    
    all_results = []
    
    # Process in batches of 50 stocks per API call
    for i in range(0, total_stocks, BATCH_SIZE):
        batch = all_stocks[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"\n📡 API CALL {batch_num}/{total_batches}")
        print(f"   Fetching {len(batch)} stocks in ONE request: {', '.join(batch[:3])}... ({len(batch)} total)")
        print("-" * 50)
        
        # ONE API CALL for 50 stocks
        batch_data = fetch_batch_stocks(batch)
        
        if batch_data:
            print(f"  ✅ Data received for {len(batch_data)} stocks")
            
            # Process each stock from the batch response
            for symbol in batch:
                if symbol in batch_data and batch_data[symbol] is not None:
                    df = batch_data[symbol]
                    if not df.empty:
                        # Save the data
                        if save_stock_data(symbol, df):
                            # Detect patterns
                            patterns = detect_patterns(df)
                            all_results.append({
                                'symbol': symbol,
                                'detected': patterns,
                                'data_points': len(df)
                            })
                            print(f"    ✅ {symbol}: {len(df)} records saved")
                        else:
                            print(f"    ❌ {symbol}: Failed to save")
                    else:
                        print(f"    ⚠️ {symbol}: No data")
                else:
                    print(f"    ⚠️ {symbol}: Not in response")
        else:
            print(f"  ❌ Batch API call failed for batch {batch_num}")
        
        # Show progress
        stocks_done = min(i + BATCH_SIZE, total_stocks)
        progress_pct = (stocks_done / total_stocks) * 100
        print(f"\n📈 Progress: {stocks_done}/{total_stocks} stocks ({progress_pct:.1f}%)")
        print(f"   ✅ Successful so far: {len(all_results)} stocks")
        
        # Save intermediate results after each batch
        os.makedirs('shape.dna', exist_ok=True)
        with open('shape.dna/patterns.json', 'w') as f:
            json.dump({'detailed_results': all_results}, f, indent=2)
        
        # Delay between API calls (except after last)
        if i + BATCH_SIZE < total_stocks:
            print(f"\n⏳ Waiting {DELAY_BETWEEN_BATCHES} seconds before next API call...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    print("\n" + "="*60)
    print(f"✅ DATA FETCHING COMPLETE!")
    print(f"   • Total API calls made: {total_batches}")
    print(f"   • Total stocks processed: {len(all_results)}/{total_stocks}")
    print(f"   • Data saved in 'data/' directory")
    
    # Generate visualizations
    if all_results:
        print("\n" + "="*60)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*60)
        
        for idx, stock_data in enumerate(all_results, 1):
            print(f"  [{idx}/{len(all_results)}] Visualizing {stock_data['symbol']}...")
            visualize_pattern(stock_data['symbol'], stock_data)
        
        print(f"\n✅ Visualizations saved in 'shape.dna/' folder")
    
    print("\n✨ ALL DONE!")

if __name__ == "__main__":
    main()
