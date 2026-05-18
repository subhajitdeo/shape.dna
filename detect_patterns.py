def main():
    detector = PredictivePatternDetector(lookback_days=60, breakout_tolerance=0.02)
    files = glob.glob('data/*.NS.json')
    
    if not files:
        print("❌ No .NS.json data files found in data/ directory.")
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
