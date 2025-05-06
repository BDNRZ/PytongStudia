
import requests
import json
import pandas as pd
import mplfinance as mpl

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

def fetch_and_display_candles(pair="BTC-USDT", interval="5m", limit=100):
    url = f'https://www.mexc.com/open/api/v2/market/kline?symbol=GLQ_USDT&interval=60m&limit=10'
    response = requests.get(url)
    response_body = response.text
    response_body_json = json.loads(response_body)
    candles_data = response_body_json["data"]
    formatted_candles_data = format_candles_data(candles_data)
    df = pd.json_normalize(formatted_candles_data)
    df.time = pd.to_datetime(df.time, unit='s')
    df = df.set_index("time")
    print(df.columns)
    print(df.head())
    mpl.plot(
        df,
        type="candle",
        title=f"Wykres świecowy dla {pair}",
        style="yahoo",
        mav=(3, 6, 9)
    )

fetch_and_display_candles()