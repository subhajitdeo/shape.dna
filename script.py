#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import os
from pathlib import Path
import yfinance as yf  # Assuming you're using yfinance for data

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
    # This is a placeholder - replace with your actual pattern detection
    patterns = []
    
    # Example pattern detection (implement your own logic here)
    close_prices = df['close'].values
    
    # Simple pattern: bullish flag (just for demonstration)
    if len(close_prices) > 10:
        recent_high = max(close_prices[-10:])
        if close_prices[-1] > recent_high * 0.95:
            patterns.append({
                'pattern': 'Bullish Flag',
                'confidence': 75
            })
    
    return patterns

def process_stock(symbol, pattern_detector=None):
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
    
    # Detect patterns (replace with your actual pattern detection)
    if pattern_detector:
        detected_patterns = pattern_detector(df)
    else:
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

def process_stocks_in_batches(stock_list, batch_size=20, delay_between_batches=5, delay_between_stocks=0.5):
    """
    Process stocks in batches to avoid API rate limiting
    
    Args:
        stock_list: List of stock symbols to process
        batch_size: Number of stocks per batch (default 20)
        delay_between_batches: Delay between batches in seconds (default 5)
        delay_between_stocks: Delay between stocks in same batch (default 0.5)
    """
    
    total_stocks = len(stock_list)
    successful = 0
    failed = 0
    results = []
    
    print(f"\n🚀 Starting to process {total_stocks} stocks in batches of {batch_size}")
    print(f"⏱️  {delay_between_batches}s delay between batches, {delay_between_stocks}s between stocks\n")
    
    for i in range(0, total_stocks, batch_size):
        batch = stock_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_stocks + batch_size - 1) // batch_size
        
        print(f"\n📦 Batch {batch_num}/{total_batches} - Processing stocks {i+1} to {min(i+batch_size, total_stocks)}")
        print("-" * 50)
        
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
            
            # Delay between stocks in same batch
            if idx < len(batch):
                time.sleep(delay_between_stocks)
        
        # Save intermediate results for this batch
        if results:
            os.makedirs('shape.dna', exist_ok=True)
            with open('shape.dna/patterns.json', 'w') as f:
                json.dump({'detailed_results': results}, f, indent=2)
            print(f"\n💾 Saved intermediate results for {len(results)} stocks")
        
        # Delay between batches (except after the last batch)
        if i + batch_size < total_stocks:
            print(f"\n⏳ Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
    
    print(f"\n✅ Processing complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Total: {total_stocks}")
    
    return results

def generate_visualizations(results, max_stocks=None):
    """Generate visualizations for processed stocks"""
    print("\n🎨 Generating visualizations...")
    
    if max_stocks:
        results = results[:max_stocks]
    
    successful_viz = 0
    for idx, stock_data in enumerate(results, 1):
        print(f"  [{idx}/{len(results)}] Visualizing {stock_data['symbol']}...")
        if visualize_pattern(stock_data['symbol'], stock_data):
            successful_viz += 1
    
    print(f"✅ Generated {successful_viz} visualizations")

def load_stocks_from_csv(csv_file='nifty500.csv'):
    """Load stock symbols from nifty500.csv"""
    try:
        df = pd.read_csv(csv_file)
        
        # Try to find the column containing stock symbols
        possible_columns = ['Symbol', 'symbol', 'Ticker', 'ticker', 'Stock', 'stock', 'Code', 'code']
        
        for col in possible_columns:
            if col in df.columns:
                stock_list = df[col].dropna().tolist()
                print(f"✅ Loaded {len(stock_list)} stocks from column '{col}' in {csv_file}")
                return stock_list
        
        # If no matching column, assume first column has symbols
        stock_list = df.iloc[:, 0].dropna().tolist()
        print(f"✅ Loaded {len(stock_list)} stocks from first column of {csv_file}")
        return stock_list
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []

def main():
    """Main execution function"""
    
    # Load stocks from CSV
    stock_list = load_stocks_from_csv('nifty500.csv')
    
    if not stock_list:
        print("❌ No stocks found in nifty500.csv")
        print("\nPlease ensure your CSV file has one of these column names:")
        print("  - Symbol, symbol, Ticker, ticker, Stock, stock, Code, code")
        print("  - Or that the first column contains stock symbols")
        return
    
    # Ask user for processing options
    print("\n" + "="*50)
    print("📈 NIFTY 500 Pattern Detection")
    print("="*50)
    print(f"Total stocks to process: {len(stock_list)}")
    print("\nProcessing options:")
    print("  1. Process all 500 stocks (may take several hours)")
    print("  2. Process first 100 stocks")
    print("  3. Process first 50 stocks")
    print("  4. Custom number")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        stocks_to_process = stock_list
    elif choice == '2':
        stocks_to_process = stock_list[:100]
    elif choice == '3':
        stocks_to_process = stock_list[:50]
    elif choice == '4':
        num = int(input("Enter number of stocks to process: "))
        stocks_to_process = stock_list[:num]
    else:
        print("Invalid choice. Processing first 50 stocks.")
        stocks_to_process = stock_list[:50]
    
    print(f"\n📊 Will process {len(stocks_to_process)} stocks")
    
    # Process stocks in batches
    results = process_stocks_in_batches(
        stock_list=stocks_to_process,
        batch_size=20,           # 20 stocks per batch
        delay_between_batches=5,  # 5 seconds between batches
        delay_between_stocks=0.5  # 0.5 seconds between stocks in same batch
    )
    
    # Generate visualizations for all processed stocks
    if results:
        generate_visualizations(results)
        
        # Print summary of detected patterns
        print("\n📊 Pattern Detection Summary:")
        print("-" * 50)
        for stock_data in results[:10]:  # Show first 10
            if stock_data['detected']:
                patterns = ", ".join([p['pattern'] for p in stock_data['detected']])
                print(f"  {stock_data['symbol']}: {patterns}")
        
        print(f"\n✨ Complete! Check 'shape.dna/' folder for visualizations")

if __name__ == "__main__":
    main()
