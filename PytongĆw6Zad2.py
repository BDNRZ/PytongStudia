import requests
import json
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt

def format_candles_data(candles_data):
    formatted_data = []
    for candle in candles_data:
        formatted_data.append({
            'time': candle[0],
            'open': float(candle[1]),
            'close': float(candle[2]),
            'high': float(candle[3]),
            'low': float(candle[4])
        })
    return formatted_data

def find_similar_pattern(df, pattern_length=10, max_deviation=0.05):
    recent_pattern = df.iloc[-pattern_length:].copy()
    
    recent_changes = np.array([
        recent_pattern['close'].pct_change().fillna(0).values,
        recent_pattern['open'].pct_change().fillna(0).values
    ]).T
    
    for i in range(len(df) - 2*pattern_length):
        historical_segment = df.iloc[i:i+pattern_length].copy()
        
        historical_changes = np.array([
            historical_segment['close'].pct_change().fillna(0).values,
            historical_segment['open'].pct_change().fillna(0).values
        ]).T
        
        mae = np.mean(np.abs(recent_changes - historical_changes))
        
        if mae < max_deviation:
            return i, i+pattern_length
    
    return None

def fetch_and_display_candles():
    url = 'https://www.mexc.com/open/api/v2/market/kline?symbol=GLQ_USDT&interval=1m&limit=500'
    
    response = requests.get(url)
    data = json.loads(response.text)["data"]
    formatted_data = format_candles_data(data)
    
    df = pd.DataFrame(formatted_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index("time")
    
    result = find_similar_pattern(df)
    
    if result:
        start_idx, end_idx = result
        similar_df = df.iloc[start_idx:end_idx]
        
        mpf.plot(
            similar_df,
            type='candle',
            title=f'Similar Pattern',
            style='yahoo',
            figsize=(10, 6)
        )
    else:
        print("No similar pattern found. Try adjusting the similarity threshold.")
        mpf.plot(
            df.iloc[-10:],
            type='candle',
            title='Recent Pattern',
            style='yahoo',
            figsize=(10, 6)
        )

if __name__ == "__main__":
    fetch_and_display_candles()