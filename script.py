#!/usr/bin/env python3
"""
Enhanced Pattern Detection with Predictive Filtering
Focus: Active, high-confidence bullish patterns (long signals only)
Outputs: shape.dna/active_patterns.json and shape.dna/active_patterns_filtered.csv
"""

import json, os, glob, logging
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
from scipy import stats
import pandas as pd
from scipy.signal import argrelextrema

# Custom JSON encoder for handling numpy types and booleans
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return int(obj)
        return super().default(obj)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictivePatternDetector:
    # Configuration constants
    MIN_DATA_POINTS = 30
    DEFAULT_LOOKBACK = 60
    BREAKOUT_TOLERANCE = 0.02
    MIN_PEAK_TROUGH_ORDER = 5
    MIN_PATTERN_POINTS = 2
    
    # Pattern-specific thresholds
    ASCENDING_PEAK_SLOPE_MAX = 0.02
    ASCENDING_TROUGH_SLOPE_MIN = 0.03
    SYMMETRICAL_PEAK_SLOPE_MAX = -0.02
    SYMMETRICAL_TROUGH_SLOPE_MIN = 0.02
    RECTANGLE_MAX_RANGE_PCT = 5.0
    RECTANGLE_MAX_VOLATILITY_PCT = 3.0
    RECTANGLE_BREAKOUT_DISTANCE_PCT = 2.0
    W_PATTERN_TROUGH_SIMILARITY_MAX = 2.0
    W_PATTERN_MIN_PEAK_HEIGHT_PCT = 3.0
    
    # Confidence and urgency weights
    VOLUME_CONFIRMATION_MULTIPLIER = 1.2
    RECTANGLE_VOLATILITY_WEIGHT = 10
    W_PATTERN_PEAK_HEIGHT_MULTIPLIER = 15
    
    def __init__(self, lookback_days=60, breakout_tolerance=0.02):
        self.lookback_days = lookback_days
        self.breakout_tolerance = breakout_tolerance
        self.rejection_reasons = {}  # Track rejections for debugging

    def load_stock_data(self, filepath):
        """Load and validate stock data from JSON file"""
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Validate required columns
            required_columns = ['time', 'close', 'high', 'low']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logger.warning(f"Missing columns {missing} in {filepath}")
                return None
            
            # Check for volume column (optional but nice)
            if 'volume' not in df.columns:
                df['volume'] = 0  # Add dummy volume column
            
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Ensure we have enough data
            if len(df) < self.MIN_DATA_POINTS:
                logger.debug(f"Insufficient data in {filepath}: {len(df)} points")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None

    def find_peaks_troughs(self, prices, order=None):
        """Find local peaks and troughs with configurable order"""
        if order is None:
            order = self.MIN_PEAK_TROUGH_ORDER
        
        if len(prices) < 2 * order + 1:
            return np.array([]), np.array([])
            
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        troughs = argrelextrema(prices, np.less, order=order)[0]
        return peaks, troughs

    def is_breakout_occurred(self, prices, breakout_level, direction='up'):
        """Check if price has already broken out beyond tolerance"""
        if len(prices) == 0:
            return True
            
        latest_close = prices[-1]
        if direction == 'up':
            return latest_close > breakout_level * (1 + self.breakout_tolerance)
        else:
            return latest_close < breakout_level * (1 - self.breakout_tolerance)

    def has_volume_confirmation(self, df, pattern_start_idx=None):
        """Check if volume is expanding during pattern formation"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return True  # No volume data, skip confirmation
        
        try:
            if pattern_start_idx is None:
                pattern_start_idx = max(0, len(df) - self.lookback_days)
            
            avg_volume = df['volume'].iloc[max(0, pattern_start_idx - 30):pattern_start_idx].mean()
            recent_volume = df['volume'].iloc[-10:].mean()
            
            if avg_volume == 0:
                return True
                
            return recent_volume > avg_volume * self.VOLUME_CONFIRMATION_MULTIPLIER
        except:
            return True  # Default to True if calculation fails

    @lru_cache(maxsize=128)
    def calculate_slope_cached(self, indices_tuple, prices_tuple):
        """Cached version of slope calculation for performance"""
        indices = np.array(indices_tuple)
        prices = np.array(prices_tuple)
        
        if len(indices) < 2:
            return 0, 0, 0, 0, 0
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(indices, prices)
        return slope, intercept, r_value, p_value, std_err

    def calculate_slope(self, indices, prices):
        """Wrapper for cached slope calculation"""
        return self.calculate_slope_cached(tuple(indices), tuple(prices))

    def detect_ascending_triangle(self, df):
        """Ascending triangle: flat resistance, rising support"""
        try:
            prices = df['close'].values[-self.lookback_days:]
            highs = df['high'].values[-self.lookback_days:]
            lows = df['low'].values[-self.lookback_days:]
            
            if len(prices) < self.MIN_DATA_POINTS:
                return None
            
            peaks, troughs = self.find_peaks_troughs(prices)
            if len(peaks) < self.MIN_PATTERN_POINTS or len(troughs) < self.MIN_PATTERN_POINTS:
                return None
            
            recent_peaks = peaks[-3:]
            recent_troughs = troughs[-3:]
            
            # Ensure we have enough points for regression
            if len(recent_peaks) < 2 or len(recent_troughs) < 2:
                return None
            
            # Resistance (peaks) – should be nearly flat
            peak_slope, peak_intercept, _, _, _ = self.calculate_slope(recent_peaks, prices[recent_peaks])
            # Support (troughs) – should be rising
            trough_slope, trough_intercept, _, _, _ = self.calculate_slope(recent_troughs, prices[recent_troughs])
            
            if abs(peak_slope) < self.ASCENDING_PEAK_SLOPE_MAX and trough_slope > self.ASCENDING_TROUGH_SLOPE_MIN:
                # Resistance level (horizontal)
                resistance_level = np.mean(prices[recent_peaks])
                # Current support line value at latest index
                support_line = trough_slope * (len(prices)-1) + trough_intercept
                
                # Check if still active (no breakout above resistance)
                if self.is_breakout_occurred(prices, resistance_level, 'up'):
                    return None  # Already broken out, not active
                
                # Calculate days to apex (when support meets resistance)
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
                
                # Volume confirmation
                volume_confirmed = self.has_volume_confirmation(df)
                volume_bonus = 15 if volume_confirmed else 0
                
                # Confidence based on slope strength and volume (simplified)
                confidence = min(100, int(trough_slope * 1000 + (1 - distance_pct/10) * 50 + volume_bonus))
                
                # Urgency score: higher when close to breakout and high confidence
                days_factor = 5 / max(1, days_to_apex) if days_to_apex else 1
                urgency = (confidence / 100) * (1 - distance_pct/10) * days_factor * 100
                
                return {
                    "pattern": "ascending_triangle",
                    "status": "active",
                    "confidence": round(confidence, 1),
                    "urgency_score": round(urgency, 1),
                    "breakout_direction": "up",
                    "breakout_level": round(resistance_level, 2),
                    "current_price": round(prices[-1], 2),
                    "distance_to_breakout_pct": round(distance_pct, 1),
                    "days_to_apex": int(days_to_apex) if days_to_apex else None,
                    "support_slope": round(trough_slope, 4),
                    "resistance_level": round(resistance_level, 2),
                    "target": round(resistance_level + (resistance_level - support_line), 2),
                    "stop_loss": round(support_line, 2),
                    "volume_confirmed": int(volume_confirmed)  # Convert boolean to int
                }
        except Exception as e:
            logger.debug(f"Ascending triangle detection error: {str(e)}")
            return None
        return None

    def detect_symmetrical_triangle(self, df):
        """Symmetrical triangle: converging support and resistance"""
        try:
            prices = df['close'].values[-self.lookback_days:]
            
            if len(prices) < self.MIN_DATA_POINTS:
                return None
            
            peaks, troughs = self.find_peaks_troughs(prices)
            if len(peaks) < self.MIN_PATTERN_POINTS or len(troughs) < self.MIN_PATTERN_POINTS:
                return None
            
            recent_peaks = peaks[-3:]
            recent_troughs = troughs[-3:]
            
            # Ensure we have enough points for regression
            if len(recent_peaks) < 2 or len(recent_troughs) < 2:
                return None
            
            peak_slope, peak_intercept, _, _, _ = self.calculate_slope(recent_peaks, prices[recent_peaks])
            trough_slope, trough_intercept, _, _, _ = self.calculate_slope(recent_troughs, prices[recent_troughs])
            
            # Converging: peak_slope negative, trough_slope positive
            if peak_slope < self.SYMMETRICAL_PEAK_SLOPE_MAX and trough_slope > self.SYMMETRICAL_TROUGH_SLOPE_MIN:
                # Find apex (intersection) with numerical stability
                denominator = trough_slope - peak_slope
                if abs(denominator) > 1e-6:  # Avoid division by zero
                    apex_x = (peak_intercept - trough_intercept) / denominator
                    apex_price = trough_slope * apex_x + trough_intercept
                    days_to_apex = max(0, apex_x - (len(prices)-1)) if apex_x > 0 else 30
                else:
                    # Parallel lines, no convergence
                    apex_x = len(prices) + 30
                    apex_price = prices[-1] * 1.05
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
                
                # Volume confirmation
                volume_confirmed = self.has_volume_confirmation(df)
                volume_bonus = 15 if volume_confirmed else 0
                
                distance_pct = (breakout_level - current_price) / current_price * 100
                distance_pct = max(0, min(distance_pct, 10))
                confidence = min(100, confidence_base + volume_bonus)
                days_factor = 5 / max(1, days_to_apex)
                urgency = (confidence/100) * (1 - distance_pct/10) * days_factor * 100
                
                return {
                    "pattern": "symmetrical_triangle",
                    "status": "active",
                    "confidence": round(confidence, 1),
                    "urgency_score": round(urgency, 1),
                    "breakout_direction": "up",
                    "breakout_level": round(breakout_level, 2),
                    "current_price": round(current_price, 2),
                    "distance_to_breakout_pct": round(distance_pct, 1),
                    "days_to_apex": int(days_to_apex) if days_to_apex else None,
                    "support_slope": round(trough_slope, 4),
                    "resistance_slope": round(peak_slope, 4),
                    "target": round(apex_price if apex_price > 0 else current_price * 1.05, 2),
                    "stop_loss": round(support_at_now, 2),
                    "volume_confirmed": int(volume_confirmed)  # Convert boolean to int
                }
        except Exception as e:
            logger.debug(f"Symmetrical triangle detection error: {str(e)}")
            return None
        return None

    def detect_rectangle(self, df):
        """Rectangle (consolidation) – bullish bias if near resistance"""
        try:
            prices = df['close'].values[-self.lookback_days:]
            highs = df['high'].values[-self.lookback_days:]
            lows = df['low'].values[-self.lookback_days:]
            
            if len(prices) < self.MIN_DATA_POINTS:
                return None
            
            # Use rolling windows to confirm consolidation
            recent_high = highs[-20:].max()
            recent_low = lows[-20:].min()
            range_size = (recent_high - recent_low) / recent_low * 100
            
            # Check if price has been range-bound (15 of last 20 days within range)
            range_days = sum(1 for h, l in zip(highs[-20:], lows[-20:]) 
                           if l >= recent_low * 0.98 and h <= recent_high * 1.02)
            
            # Volatility check
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if np.mean(prices[-20:]) > 0 else 100
            
            if range_size < self.RECTANGLE_MAX_RANGE_PCT and range_days >= 15 and volatility < self.RECTANGLE_MAX_VOLATILITY_PCT:
                # Check if price is near the top (bullish)
                current_price = prices[-1]
                distance_to_res = (recent_high - current_price) / current_price * 100
                
                if distance_to_res < self.RECTANGLE_BREAKOUT_DISTANCE_PCT:  # within 2% of resistance -> potential breakout
                    breakout_level = recent_high
                    if self.is_breakout_occurred(prices, breakout_level, 'up'):
                        return None
                    
                    # Volume confirmation
                    volume_confirmed = self.has_volume_confirmation(df)
                    volume_bonus = 15 if volume_confirmed else 0
                    
                    confidence = min(100, (1 - volatility/self.RECTANGLE_VOLATILITY_WEIGHT) * 100 + volume_bonus)
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
                        "stop_loss": round(recent_low, 2),
                        "volume_confirmed": int(volume_confirmed),  # Convert boolean to int
                        "range_days": range_days
                    }
        except Exception as e:
            logger.debug(f"Rectangle detection error: {str(e)}")
            return None
        return None

    def detect_w_pattern(self, df):
        """Double bottom (W) – bullish reversal"""
        try:
            prices = df['close'].values[-self.lookback_days:]
            
            if len(prices) < self.MIN_DATA_POINTS:
                return None
            
            peaks, troughs = self.find_peaks_troughs(prices, order=3)
            
            if len(troughs) >= 2 and len(peaks) >= 1:
                last_two_troughs = troughs[-2:]
                peak_between = peaks[(peaks > last_two_troughs[0]) & (peaks < last_two_troughs[1])]
                
                if len(peak_between) >= 1:
                    t1_price = prices[last_two_troughs[0]]
                    t2_price = prices[last_two_troughs[1]]
                    neckline_price = prices[peak_between[0]]
                    
                    trough_similarity = abs(t1_price - t2_price) / t1_price * 100 if t1_price > 0 else 100
                    peak_height = (neckline_price - t1_price) / t1_price * 100 if t1_price > 0 else 0
                    
                    if trough_similarity < self.W_PATTERN_TROUGH_SIMILARITY_MAX and peak_height > self.W_PATTERN_MIN_PEAK_HEIGHT_PCT:
                        current_price = prices[-1]
                        # Check if price has already broken above neckline (confirmed)
                        if current_price > neckline_price:
                            return None  # Already broken, not active (we want before breakout)
                        
                        distance_to_breakout = (neckline_price - current_price) / current_price * 100 if current_price > 0 else 100
                        if distance_to_breakout < 0:
                            return None
                        
                        # Volume confirmation (often high volume on the second bottom)
                        volume_confirmed = self.has_volume_confirmation(df)
                        volume_bonus = 15 if volume_confirmed else 0
                        
                        confidence = min(100, peak_height * self.W_PATTERN_PEAK_HEIGHT_MULTIPLIER + volume_bonus)
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
                            "stop_loss": round(t1_price * 0.98, 2),
                            "volume_confirmed": int(volume_confirmed),  # Convert boolean to int
                            "trough_similarity": round(trough_similarity, 2)
                        }
        except Exception as e:
            logger.debug(f"W pattern detection error: {str(e)}")
            return None
        return None

    def detect_all_bullish_patterns(self, filepath):
        """Run all detectors and return active patterns only"""
        df = self.load_stock_data(filepath)
        if df is None:
            return None
        
        symbol = os.path.basename(filepath).replace('.json', '')
        patterns = []
        
        detectors = [
            self.detect_ascending_triangle,
            self.detect_symmetrical_triangle, 
            self.detect_rectangle,
            self.detect_w_pattern
        ]
        
        for detector in detectors:
            try:
                pat = detector(df)
                if pat:
                    patterns.append(pat)
            except Exception as e:
                logger.debug(f"Detector {detector.__name__} failed for {symbol}: {str(e)}")
        
        if not patterns:
            # Track rejection reason for debugging
            self.rejection_reasons[symbol] = "No patterns detected"
            return None
        
        # Return the best pattern (highest urgency) per symbol
        best = max(patterns, key=lambda x: x['urgency_score'])
        best['symbol'] = symbol
        best['date'] = df.index[-1].strftime('%Y-%m-%d')
        return best

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    detector = PredictivePatternDetector(lookback_days=60, breakout_tolerance=0.02)
    files = glob.glob('data/*.json')
    
    if not files:
        print("❌ No data files found. Run fetch first.")
        logger.error("No data files found in data/ directory")
        return
    
    print(f"🔍 Analyzing {len(files)} stocks for active bullish patterns...")
    logger.info(f"Starting analysis of {len(files)} files")
    
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
    
    # Save JSON with full details using custom encoder
    output_json = {
        "scan_date": datetime.now().isoformat(),
        "total_stocks_scanned": len(files),
        "active_patterns_count": len(active_patterns),
        "patterns": active_patterns
    }
    
    with open('shape.dna/active_patterns.json', 'w') as fp:
        json.dump(output_json, fp, indent=2, cls=CustomJSONEncoder)  # Use custom encoder
    
    # Save filtered CSV (high confidence and high urgency)
    high_confidence = [p for p in active_patterns if p['confidence'] >= 70 and p['urgency_score'] >= 50]
    
    if high_confidence:
        df = pd.DataFrame(high_confidence)
        # Select relevant columns for CSV
        available_cols = ['symbol', 'date', 'pattern', 'confidence', 'urgency_score', 
                         'breakout_direction', 'breakout_level', 'current_price', 
                         'distance_to_breakout_pct', 'days_to_apex', 'target', 'stop_loss']
        
        # Filter to only columns that exist
        cols_to_save = [col for col in available_cols if col in df.columns]
        df[cols_to_save].to_csv('shape.dna/active_patterns_high_confidence.csv', index=False)
        
        print(f"\n✅ Found {len(high_confidence)} high-confidence active patterns.")
        print(f"   Saved to shape.dna/active_patterns_high_confidence.csv")
        
        # Print top 5 patterns
        print("\n🏆 TOP 5 HIGHEST URGENCY PATTERNS:")
        for i, pattern in enumerate(high_confidence[:5], 1):
            print(f"   {i}. {pattern['symbol']} - {pattern['pattern']} (Urgency: {pattern['urgency_score']}, Confidence: {pattern['confidence']})")
    else:
        print("\n⚠️ No high-confidence active patterns found today.")
    
    # Save rejection summary for debugging
    if detector.rejection_reasons:
        rejection_summary = {
            "total_rejected": len(detector.rejection_reasons),
            "reasons": detector.rejection_reasons
        }
        with open('shape.dna/rejected_patterns.json', 'w') as fp:
            json.dump(rejection_summary, fp, indent=2, cls=CustomJSONEncoder)
        print(f"\n📝 Rejection summary saved to shape.dna/rejected_patterns.json")
    
    print(f"\n📊 Summary: {len(active_patterns)} total active patterns detected.")
    print(f"📁 Full data: shape.dna/active_patterns.json")
    
    # Performance metrics
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
