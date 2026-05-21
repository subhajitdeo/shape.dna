#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import os
from pathlib import Path
import yfinance as yf

def fetch_stock_data(symbol, period='3mo'):
    """Fetch stock data for a given symbol"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            print(f"  ❌ No data found for {symbol}")
            return None
        
        # Convert to format expected by your visualization
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
        
        # Save individual stock data
        os.makedirs('data', exist_ok=True)
        with open(f'data/{symbol}.json', 'w') as f:
            json.dump(data, f)
        
        return data
    except Exception as e:
        print(f"  ❌ Error fetching {symbol}: {e}")
        return None

def detect_patterns(df):
    """Simple pattern detection - replace with your actual pattern detection logic"""
    patterns = []
    
    # Your existing pattern detection logic here
    # This is just a placeholder
    close_prices = df['close'].values
    
    if len(close_prices) > 10:
        recent_high = max(close_prices[-10:])
        if close_prices[-1] > recent_high * 0.95:
            patterns.append({
                'pattern': 'Bullish Flag',
                'confidence': 75
            })
    
    return patterns

def process_stock(symbol):
    """Process a single stock: fetch data and detect patterns"""
    print(f"  📊 Fetching data for {symbol}...")
    
    # Fetch stock data
    data = fetch_stock_data(symbol)
    if not data:
        return None
    
    # Create dataframe for pattern detection
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Detect patterns
    detected_patterns = detect_patterns(df)
    
    return {
        'symbol': symbol,
        'detected': detected_patterns,
        'data_points': len(data)
    }

def visualize_pattern(symbol, pattern_data):
    """Visualize a detected pattern for a specific stock"""
    
    # Load stock data
    data_file = f'data/{symbol}.json'
    if not os.path.exists(data_file):
        print(f"⚠️ Data file not found for {symbol}, skipping...")
        return False
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Plot recent data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-60:], df['close'].values[-60:], 'b-', label='Close Price', linewidth=1.5)
        
        # Add pattern annotation
        if pattern_data['detected']:
            pattern_text = "\n".join([f"{p['pattern']}: {p['confidence']}%" 
                                       for p in pattern_data['detected']])
        else:
            pattern_text = "No patterns detected"
        
        ax.text(0.02, 0.98, pattern_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_title(f"{symbol} - Chart Patterns Detected")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Ensure directory exists
        Path('shape.dna').mkdir(exist_ok=True)
        plt.savefig(f'shape.dna/{symbol}_pattern.png', dpi=100)
        plt.close()
        
        return True
    except Exception as e:
        print(f"❌ Error visualizing {symbol}: {e}")
        return False

def process_stocks_in_batches(stock_list, batch_size=20, delay_between_batches=5):
    """
    Process stocks in batches to avoid API rate limiting
    
    Args:
        stock_list: List of stock symbols to process
        batch_size: Number of stocks per batch (default 20)
        delay_between_batches: Delay between batches in seconds (default 5)
    """
    
    total_stocks = len(stock_list)
    successful = 0
    failed = 0
    results = []
    
    # Calculate number of batches
    num_batches = (total_stocks + batch_size - 1) // batch_size
    
    print(f"\n🚀 Starting to process {total_stocks} stocks")
    print(f"📦 Batch size: {batch_size} stocks per batch")
    print(f"📊 Total batches: {num_batches}")
    print(f"⏱️  Delay between batches: {delay_between_batches} seconds\n")
    print("=" * 60)
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_stocks)
        batch = stock_list[start_idx:end_idx]
        
        print(f"\n📦 BATCH {batch_num + 1}/{num_batches}")
        print(f"   Stocks {start_idx + 1} to {end_idx} of {total_stocks}")
        print(f"   Processing {len(batch)} stocks in this batch")
        print("-" * 40)
        
        # Process each stock in the current batch
        for idx, symbol in enumerate(batch, 1):
            print(f"  [{idx}/{len(batch)}] Processing {symbol}...")
            
            # Process the stock
            result = process_stock(symbol)
            
            if result:
                successful += 1
                results.append(result)
                print(f"    ✅ Successfully fetched data for {symbol}")
            else:
                failed += 1
                print(f"    ❌ Failed to fetch data for {symbol}")
            
            # Small delay between stocks in same batch (0.5 seconds)
            if idx < len(batch):
                time.sleep(0.5)
        
        # Save intermediate results after each batch
        if results:
            os.makedirs('shape.dna', exist_ok=True)
            with open('shape.dna/patterns.json', 'w') as f:
                json.dump({'detailed_results': results}, f, indent=2)
            print(f"\n💾 Saved intermediate results ({len(results)} stocks so far)")
        
        # Show progress
        progress_pct = (end_idx / total_stocks) * 100
        print(f"\n📈 Progress: {end_idx}/{total_stocks} stocks processed ({progress_pct:.1f}%)")
        
        # Delay between batches (except after the last batch)
        if batch_num + 1 < num_batches:
            print(f"\n⏳ Waiting {delay_between_batches} seconds before next batch...")
            print("   (This prevents API rate limiting)")
            time.sleep(delay_between_batches)
    
    print("\n" + "=" * 60)
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total: {total_stocks}")
    print(f"   📁 Data saved in 'data/' directory")
    print(f"   📈 Patterns saved in 'shape.dna/patterns.json'")
    
    return results

def load_stocks_from_csv(csv_file='nifty500.csv'):
    """Load stock symbols from nifty500.csv"""
    try:
        df = pd.read_csv(csv_file)
        
        # Try to find the column containing stock symbols
        possible_columns = ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock', 'Code', 'code']
        
        for col in possible_columns:
            if col in df.columns:
                stock_list = df[col].dropna().tolist()
                # Clean up symbols (remove any whitespace)
                stock_list = [str(s).strip().upper() for s in stock_list]
                print(f"✅ Loaded {len(stock_list)} stocks from column '{col}' in {csv_file}")
                return stock_list
        
        # If no matching column, assume first column has symbols
        stock_list = df.iloc[:, 0].dropna().tolist()
        stock_list = [str(s).strip().upper() for s in stock_list]
        print(f"✅ Loaded {len(stock_list)} stocks from first column of {csv_file}")
        return stock_list
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("📈 NIFTY 500 PATTERN DETECTION SYSTEM")
    print("="*60)
    
    # Load stocks from CSV
    stock_list = load_stocks_from_csv('nifty500.csv')
    
    if not stock_list:
        print("❌ No stocks found in nifty500.csv")
        print("\nPlease ensure your CSV file exists and has stock symbols")
        return
    
    # Display first few stocks
    print(f"\n📋 First 10 stocks from your list:")
    for i, symbol in enumerate(stock_list[:10], 1):
        print(f"   {i}. {symbol}")
    
    # Ask user for processing options
    print("\n" + "-"*60)
    print("Processing Options:")
    print("-"*60)
    print(f"   Total stocks available: {len(stock_list)}")
    print(f"   Batch size: 20 stocks per batch")
    print(f"   Delay between batches: 5 seconds")
    print(f"\n   Estimated time for:")
    print(f"   - 100 stocks: ~{(100/20)*5 + 100*0.5:.0f} minutes")
    print(f"   - 500 stocks: ~{(500/20)*5 + 500*0.5:.0f} minutes")
    print("-"*60)
    
    choice = input("\nHow many stocks do you want to process?\n  [1] All 500 stocks\n  [2] First 100 stocks\n  [3] First 50 stocks\n  [4] Custom number\n\nChoice (1-4): ").strip()
    
    if choice == '1':
        stocks_to_process = stock_list
    elif choice == '2':
        stocks_to_process = stock_list[:100]
        print(f"\n⚠️  Note: You'll process {len(stocks_to_process)} stocks in 5 batches of 20")
    elif choice == '3':
        stocks_to_process = stock_list[:50]
        print(f"\n⚠️  Note: You'll process {len(stocks_to_process)} stocks in 3 batches of 20")
    elif choice == '4':
        num = int(input("Enter number of stocks to process: "))
        stocks_to_process = stock_list[:num]
        num_batches = (num + 19) // 20  # Ceiling division
        print(f"\n⚠️  Note: You'll process {len(stocks_to_process)} stocks in {num_batches} batches of 20")
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    confirm = input(f"\n✅ Ready to process {len(stocks_to_process)} stocks in batches of 20. Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled.")
        return
    
    # Process stocks in batches of 20 with 5 second delay
    results = process_stocks_in_batches(
        stock_list=stocks_to_process,
        batch_size=20,           # ← 20 stocks per batch as you requested
        delay_between_batches=5  # ← 5 seconds between batches
    )
    
    # Generate visualizations for processed stocks
    if results:
        print("\n" + "="*60)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*60)
        
        successful_viz = 0
        for idx, stock_data in enumerate(results, 1):
            print(f"  [{idx}/{len(results)}] Visualizing {stock_data['symbol']}...")
            if visualize_pattern(stock_data['symbol'], stock_data):
                successful_viz += 1
        
        print(f"\n✅ Generated {successful_viz} visualizations in 'shape.dna/' folder")
        
        # Print summary of detected patterns
        stocks_with_patterns = [r for r in results if r['detected']]
        if stocks_with_patterns:
            print(f"\n📊 Pattern Detection Summary:")
            print("-" * 50)
            print(f"   Found patterns in {len(stocks_with_patterns)} out of {len(results)} stocks")
            print(f"\n   Top stocks with patterns:")
            for stock_data in stocks_with_patterns[:10]:
                patterns = ", ".join([p['pattern'] for p in stock_data['detected']])
                print(f"   • {stock_data['symbol']}: {patterns}")
    
    print("\n✨ ALL DONE! Check the following folders:")
    print(f"   📁 'data/' - Raw stock data (JSON files)")
    print(f"   📁 'shape.dna/' - Pattern visualizations (PNG files)")
    print(f"   📄 'shape.dna/patterns.json' - Pattern detection results")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
