#!/usr/bin/env python3
import json, os, glob
from datetime import datetime
import numpy as np
from scipy import stats
import pandas as pd
from scipy.signal import argrelextrema

class PatternDetector:
    def __init__(self, lookback_days=60, min_pattern_days=20):
        self.lookback_days = lookback_days
        self.min_pattern_days = min_pattern_days

    def load_stock_data(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df

    def find_peaks_troughs(self, prices, order=5):
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        troughs = argrelextrema(prices, np.less, order=order)[0]
        return peaks, troughs

    def detect_triangle(self, df):
        prices = df['close'].values[-self.lookback_days:]
        peaks, troughs = self.find_peaks_troughs(prices)
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        recent_peaks = peaks[-3:]
        recent_troughs = troughs[-3:]
        peak_slope, _, _, _, _ = stats.linregress(recent_peaks, prices[recent_peaks])
        trough_slope, _, _, _, _ = stats.linregress(recent_troughs, prices[recent_troughs])
        if peak_slope < -0.05 and trough_slope > 0.05:
            return {"pattern":"symmetrical_triangle","confidence":round(min(abs(peak_slope),trough_slope)*100,2),"current_price":round(prices[-1],2)}
        elif peak_slope < -0.05 and abs(trough_slope) < 0.02:
            return {"pattern":"descending_triangle","confidence":round(abs(peak_slope)*100,2),"current_price":round(prices[-1],2)}
        elif trough_slope > 0.05 and abs(peak_slope) < 0.02:
            return {"pattern":"ascending_triangle","confidence":round(trough_slope*100,2),"current_price":round(prices[-1],2)}
        return None

    def detect_rectangle(self, df):
        prices = df['close'].values[-self.lookback_days:]
        highs = df['high'].values[-self.lookback_days:]
        lows = df['low'].values[-self.lookback_days:]
        recent_high = np.percentile(highs, 95)
        recent_low = np.percentile(lows, 5)
        range_size = (recent_high - recent_low) / recent_low * 100
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100
        if range_size < 5 and volatility < 3:
            return {"pattern":"rectangle","confidence":round((1-volatility/10)*100,2),"resistance":round(recent_high,2),"support":round(recent_low,2),"current_price":round(prices[-1],2)}
        return None

    def detect_m_pattern(self, df):
        prices = df['close'].values[-self.lookback_days:]
        peaks, troughs = self.find_peaks_troughs(prices, order=3)
        if len(peaks) >= 2 and len(troughs) >= 1:
            last_two_peaks = peaks[-2:]
            trough_between = troughs[(troughs > last_two_peaks[0]) & (troughs < last_two_peaks[1])]
            if len(trough_between) >= 1:
                p1, p2 = prices[last_two_peaks[0]], prices[last_two_peaks[1]]
                trough_price = prices[trough_between[0]]
                if abs(p1-p2)/p1*100 < 2 and (p1-trough_price)/p1*100 > 3:
                    return {"pattern":"m_pattern_double_top","confidence":round(min(100,(p1-trough_price)/p1*15),2),"peak1":round(p1,2),"peak2":round(p2,2),"neckline":round(trough_price,2),"current_price":round(prices[-1],2)}
        return None

    def detect_w_pattern(self, df):
        prices = df['close'].values[-self.lookback_days:]
        peaks, troughs = self.find_peaks_troughs(prices, order=3)
        if len(troughs) >= 2 and len(peaks) >= 1:
            last_two_troughs = troughs[-2:]
            peak_between = peaks[(peaks > last_two_troughs[0]) & (peaks < last_two_troughs[1])]
            if len(peak_between) >= 1:
                t1, t2 = prices[last_two_troughs[0]], prices[last_two_troughs[1]]
                peak_price = prices[peak_between[0]]
                if abs(t1-t2)/t1*100 < 2 and (peak_price-t1)/t1*100 > 3:
                    return {"pattern":"w_pattern_double_bottom","confidence":round(min(100,(peak_price-t1)/t1*15),2),"bottom1":round(t1,2),"bottom2":round(t2,2),"neckline":round(peak_price,2),"current_price":round(prices[-1],2)}
        return None

    def detect_all_patterns(self, filepath):
        try:
            df = self.load_stock_data(filepath)
            if len(df) < self.min_pattern_days:
                return None
            symbol = os.path.basename(filepath).replace('.json','')
            detected = [p for p in [self.detect_triangle(df), self.detect_rectangle(df), self.detect_m_pattern(df), self.detect_w_pattern(df)] if p]
            if detected:
                return {"symbol":symbol,"date":df.index[-1].strftime('%Y-%m-%d'),"current_price":round(df['close'].iloc[-1],2),"detected":detected}
        except Exception as e:
            print(f"Error {filepath}: {e}")
        return None

def main():
    detector = PatternDetector()
    files = glob.glob('data/*.json')
    if not files:
        print("❌ No data/*.json files found")
        return
    print(f"🔍 Scanning {len(files)} files...")
    results = []
    by_type = {'triangles':0,'rectangles':0,'m_patterns':0,'w_patterns':0}
    for i, f in enumerate(files,1):
        if i%50==0: print(f"Progress: {i}/{len(files)}")
        r = detector.detect_all_patterns(f)
        if r:
            results.append(r)
            for p in r['detected']:
                if 'triangle' in p['pattern']: by_type['triangles']+=1
                elif 'rectangle' in p['pattern']: by_type['rectangles']+=1
                elif 'm_pattern' in p['pattern']: by_type['m_patterns']+=1
                elif 'w_pattern' in p['pattern']: by_type['w_patterns']+=1
    os.makedirs('shape.dna', exist_ok=True)
    out = {"scan_date":datetime.now().isoformat(),"total_stocks_scanned":len(files),"stocks_with_patterns":len(results),"patterns_by_type":by_type,"detailed_results":results}
    with open('shape.dna/patterns.json','w') as fp:
        json.dump(out, fp, indent=2)
    pd.DataFrame([{'symbol':s['symbol'],'date':s['date'],'current_price':s['current_price'],'pattern_type':p['pattern'],'confidence':p['confidence']} for s in results for p in s['detected']]).to_csv('shape.dna/patterns_summary.csv', index=False)
    print(f"\n✅ Done. Found patterns in {len(results)} stocks. Results in shape.dna/")

if __name__=="__main__":
    main()
