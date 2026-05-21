#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def visualize_pattern(symbol, pattern_data):
    """Visualize a detected pattern for a specific stock"""
    
    # Load stock data
    with open(f'data/{symbol}.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Plot recent data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-60:], df['close'].values[-60:], 'b-', label='Close Price', linewidth=1.5)
    
    # Add pattern annotation
    pattern_text = "\n".join([f"{p['pattern']}: {p['confidence']}%" 
                               for p in pattern_data['detected']])
    
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
    plt.savefig(f'shape.dna/{symbol}_pattern.png', dpi=100)
    plt.close()

# Load patterns and visualize
with open('shape.dna/patterns.json', 'r') as f:
    patterns = json.load(f)

for stock in patterns['detailed_results'][:5]:  # Visualize top 5
    visualize_pattern(stock['symbol'], stock)
    print(f"📊 Generated chart for {stock['symbol']}")
