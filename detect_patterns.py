#!/usr/bin/env python3
"""
Enhanced Pattern Detection with Predictive Filtering
Focus: Active, high-confidence bullish patterns (long signals only)
Outputs: shape.dna/active_patterns.json and shape.dna/active_patterns_filtered.csv
"""

import json, os, glob
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from scipy.signal import argrelextrema

class PredictivePatternDetector:
    def __init__(self, lookback_days=60, breakout_tolerance=0.02):
        self.lookback_days = lookback_days
        self.breakout_tolerance = breakout_tolerance  # 2% tolerance for breakout detection

    def load_stock_data(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        # Ensure we have enough data
        if len(df) < 30:
            return None
        return df

    def find_peaks_troughs(self, prices, order=5):
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        troughs = argrelextrema(prices, np.less, order=order)[0]
        return peaks, troughs

    def is_breakout_occurred(self, prices, breakout_level, direction='up'):
        """Check if price has already broken out beyond tolerance"""
        latest_close = prices[-1]
        if direction == 'up':
            return latest_close > breakout_level * (1 + self.breakout_tolerance)
        else:
            return latest_close < breakout_level * (1 - self.breakout_tolerance)

    def detect_ascending_triangle(self, df):
        """Ascending triangle: flat resistance, rising support"""
        prices = df['close'].values[-self.lookback_days:]
        highs = df['high'].values[-self.lookback_days:]
        lows = df['low'].values[-self.lookback_days:]
        
        peaks, troughs = self.find_peaks_troughs(prices)
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        recent_peaks = peaks[-3:]
        recent_troughs = troughs[-3:]
        
        # Resistance (peaks) – should be nearly flat
        peak_slope, peak_intercept, _, _, _ = stats.linregress(recent_peaks, prices[recent_peaks])
        # Support (troughs) – should be rising
        trough_slope, trough_intercept, _, _, _ = stats.linregress(recent_troughs, prices[recent_troughs])
        
        if abs(peak_slope) < 0.02 and trough_slope > 0.03:
            # Resistance level (horizontal)
            resistance_level = np.mean(prices[recent_peaks])
            # Current support line value at latest index
            support_line = trough_slope * (len(prices)-1) + trough_intercept
            
            # Check if still active (no breakout above resistance)
            if self.is_breakout_occurred(prices, resistance_level, 'up'):
                return None  # Already broken out, not active
            
            # Calculate days to apex (when support meets resistance)
            # Solve: trough_slope * x + trough_intercept = resistance_level
            if trough_slope > 0:
                days_to_apex = (resistance_level - trough_intercept) / trough_slope
                days_to_apex = max(0, days_to_apex - (len(prices)-1))
            else:
                days_to_apex = 30  # fallback
            
            # Distance to breakout (as %)
            distance_pct = (resistance_level - prices[-1]) / prices[-1] * 100
            if distance_pct < 0:
                distance_pct = 0  # already above resistance? then should have been filtered
            distance_pct = min(distance_pct, 10)
            
            # Confidence based on slope strength and volume (simplified)
            confidence = min(100, int(trough_slope * 1000 + (1 - distance_pct/10) * 50))
            
            # Urgency score: higher when close to breakout and high confidence
            urgency = (confidence / 100) * (1 - distance_pct/10) * (5 / max(1, days_to_apex)) * 100
            
            return {
                "pattern": "ascending_triangle",
                "status": "active",
                "confidence": round(confidence, 1),
                "urgency_score": round(urgency, 1),
                "breakout_direction": "up",
                "breakout_level": round(resistance_level, 2),
                "current_price": round(prices[-1], 2),
                "distance_to_breakout_pct": round(distance_pct, 1),
                "days_to_apex": int(days_to_apex),
                "support_slope": round(trough_slope, 4),
                "resistance_level": round(resistance_level, 2),
                "target": round(resistance_level + (resistance_level - support_line), 2),
                "stop_loss": round(support_line, 2)
            }
        return None

    def detect_symmetrical_triangle(self, df):
        """Symmetrical triangle: converging support and resistance"""
        prices = df['close'].values[-self.lookback_days:]
        
        peaks, troughs = self.find_peaks_troughs(prices)
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        recent_peaks = peaks[-3:]
        recent_troughs = troughs[-3:]
        
        peak_slope, peak_intercept, _, _, _ = stats.linregress(recent_peaks, prices[recent_peaks])
        trough_slope, trough_intercept, _, _, _ = stats.linregress(recent_troughs, prices[recent_troughs])
        
        # Converging: peak_slope negative, trough_slope positive
        if peak_slope < -0.02 and trough_slope > 0.02:
            # Find apex (intersection)
            if trough_slope - peak_slope != 0:
                apex_x = (peak_intercept - trough_intercept) / (trough_slope - peak_slope)
                apex_price = trough_slope * apex_x + trough_intercept
                days_to_apex = max(0, apex_x - (len(prices)-1))
            else:
                days_to_apex = 30
            
            # Determine breakout direction based on which line is closer
            current_price = prices[-1]
            resistance_at_now = peak_slope * (len(prices)-1) + peak_intercept
            support_at_now = trough_slope * (len(prices)-1) + trough_intercept
            distance_to_res = (resistance_at_now - current_price) / current_price * 100
            distance_to_sup = (current_price - support_at_now) / current_price * 100
            
            # For long bias, we prefer breakout upward if price is closer to support (bouncing)
            if distance_to_sup < distance_to_res:
                breakout_direction = "up"
                breakout_level = resistance_at_now
                confidence_base = (1 - distance_to_res/10) * 100
            else:
                # Not a strong long signal, skip (we want only long)
                return None
            
            if self.is_breakout_occurred(prices, breakout_level, 'up'):
                return None
            
            distance_pct = (breakout_level - current_price) / current_price * 100
            distance_pct = max(0, min(distance_pct, 10))
            confidence = min(100, confidence_base)
            urgency = (confidence/100) * (1 - distance_pct/10) * (5 / max(1, days_to_apex)) * 100
            
            return {
                "pattern": "symmetrical_triangle",
                "status": "active",
                "confidence": round(confidence, 1),
                "urgency_score": round(urgency, 1),
                "breakout_direction": "up",
                "breakout_level": round(breakout_level, 2),
                "current_price": round(current_price, 2),
                "distance_to_breakout_pct": round(distance_pct, 1),
                "days_to_apex": int(days_to_apex),
                "support_slope": round(trough_slope, 4),
                "resistance_slope": round(peak_slope, 4),
                "target": round(apex_price if apex_price > 0 else current_price * 1.05, 2),
                "stop_loss": round(support_at_now, 2)
            }
        return None

    def detect_rectangle(self, df):
        """Rectangle (consolidation) – bullish bias if near resistance"""
        prices = df['close'].values[-self.lookback_days:]
        highs = df['high'].values[-self.lookback_days:]
        lows = df['low'].values[-self.lookback_days:]
        
        # Calculate range boundaries
        recent_high = np.percentile(highs, 95)
        recent_low = np.percentile(lows, 5)
        range_size = (recent_high - recent_low) / recent_low * 100
        
        # Volatility check
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100
        
        if range_size < 5 and volatility < 3:
            # Check if price is near the top (bullish)
            current_price = prices[-1]
            distance_to_res = (recent_high - current_price) / current_price * 100
            
            if distance_to_res < 2:  # within 2% of resistance -> potential breakout
                breakout_level = recent_high
                if self.is_breakout_occurred(prices, breakout_level, 'up'):
                    return None
                
                confidence = min(100, (1 - volatility/10) * 100)
                # For rectangle, "days to apex" is not defined, set high urgency if near resistance
                urgency = (confidence/100) * (1 - distance_to_res/5) * 100
                
                return {
                    "pattern": "rectangle",
                    "status": "active",
                    "confidence": round(confidence, 1),
                    "urgency_score": round(urgency, 1),
                    "breakout_direction": "up",
                    "breakout_level": round(breakout_level, 2),
                    "current_price": round(current_price, 2),
                    "distance_to_breakout_pct": round(distance_to_res, 1),
                    "days_to_apex": None,  # Not applicable
                    "support": round(recent_low, 2),
                    "resistance": round(recent_high, 2),
                    "target": round(recent_high + (recent_high - recent_low), 2),
                    "stop_loss": round(recent_low, 2)
                }
        return None

    def detect_w_pattern(self, df):
        """Double bottom (W) – bullish reversal"""
        prices = df['close'].values[-self.lookback_days:]
        peaks, troughs = self.find_peaks_troughs(prices, order=3)
        
        if len(troughs) >= 2 and len(peaks) >= 1:
            last_two_troughs = troughs[-2:]
            peak_between = peaks[(peaks > last_two_troughs[0]) & (peaks < last_two_troughs[1])]
            
            if len(peak_between) >= 1:
                t1_price = prices[last_two_troughs[0]]
                t2_price = prices[last_two_troughs[1]]
                neckline_price = prices[peak_between[0]]
                
                trough_similarity = abs(t1_price - t2_price) / t1_price * 100
                peak_height = (neckline_price - t1_price) / t1_price * 100
                
                if trough_similarity < 2 and peak_height > 3:
                    current_price = prices[-1]
                    # Check if price has already broken above neckline (confirmed)
                    if current_price > neckline_price:
                        return None  # Already broken, not active (we want before breakout)
                    
                    distance_to_breakout = (neckline_price - current_price) / current_price * 100
                    if distance_to_breakout < 0:
                        return None
                    
                    confidence = min(100, peak_height * 15)
                    urgency = (confidence/100) * (1 - distance_to_breakout/10) * 100
                    
                    return {
                        "pattern": "w_pattern_double_bottom",
                        "status": "active",
                        "confidence": round(confidence, 1),
                        "urgency_score": round(urgency, 1),
                        "breakout_direction": "up",
                        "breakout_level": round(neckline_price, 2),
                        "current_price": round(current_price, 2),
                        "distance_to_breakout_pct": round(distance_to_breakout, 1),
                        "days_to_apex": None,
                        "neckline": round(neckline_price, 2),
                        "target": round(neckline_price + (neckline_price - t1_price), 2),
                        "stop_loss": round(t1_price * 0.98, 2)
                    }
        return None

    def detect_all_bullish_patterns(self, filepath):
        """Run all detectors and return active patterns only"""
        df = self.load_stock_data(filepath)
        if df is None:
            return None
        
        symbol = os.path.basename(filepath).replace('.json', '')
        patterns = []
        
        for detector in [self.detect_ascending_triangle, self.detect_symmetrical_triangle, 
                         self.detect_rectangle, self.detect_w_pattern]:
            pat = detector(df)
            if pat:
                patterns.append(pat)
        
        if not patterns:
            return None
        
        # Return the best pattern (highest urgency) per symbol
        best = max(patterns, key=lambda x: x['urgency_score'])
        best['symbol'] = symbol
        best['date'] = df.index[-1].strftime('%Y-%m-%d')
        return best

def main():
    detector = PredictivePatternDetector(lookback_days=60, breakout_tolerance=0.02)
    files = glob.glob('data/*.json')
    
    if not files:
        print("❌ No data files found. Run fetch first.")
        return
    
    print(f"🔍 Analyzing {len(files)} stocks for active bullish patterns...")
    active_patterns = []
    
    for i, f in enumerate(files, 1):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(files)}")
        result = detector.detect_all_bullish_patterns(f)
        if result:
            active_patterns.append(result)
    
    # Sort by urgency score (highest first)
    active_patterns.sort(key=lambda x: x['urgency_score'], reverse=True)
    
    # Save full results
    os.makedirs('shape.dna', exist_ok=True)
    with open('shape.dna/active_patterns.json', 'w') as fp:
        json.dump({
            "scan_date": datetime.now().isoformat(),
            "total_stocks_scanned": len(files),
            "active_patterns_count": len(active_patterns),
            "patterns": active_patterns
        }, fp, indent=2)
    
    # Save filtered CSV (only high confidence > 70, urgency > 50)
    high_confidence = [p for p in active_patterns if p['confidence'] >= 70 and p['urgency_score'] >= 50]
    if high_confidence:
        df = pd.DataFrame(high_confidence)
        # Select relevant columns
        cols = ['symbol', 'date', 'pattern', 'confidence', 'urgency_score', 
                'breakout_direction', 'breakout_level', 'current_price', 
                'distance_to_breakout_pct', 'days_to_apex', 'target', 'stop_loss']
        df[cols].to_csv('shape.dna/active_patterns_high_confidence.csv', index=False)
        print(f"\n✅ Found {len(high_confidence)} high-confidence active patterns.")
        print(f"   Saved to shape.dna/active_patterns_high_confidence.csv")
    else:
        print("\n⚠️ No high-confidence active patterns found today.")
    
    print(f"\n📊 Summary: {len(active_patterns)} total active patterns detected.")
    print(f"📁 Full data: shape.dna/active_patterns.json")

if __name__ == "__main__":
    main()
