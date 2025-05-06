import requests  # biblioteka ta służy do wykonywania zapytań (pobierania danych z internetu)
import json  # biblioteka ta przetwarza dane w formacie json, tak aby można je było dalej obrabiać w kodzie
import pandas as pd  # ta biblioteka pozwala na przetwarzanie zbiorów danych
import mplfinance as mpl  # biblioteka do rysowania wykresów finansowych

def format_candles_data(candles_data):
    """
    Funkcja formatująca dane świeczek do odpowiedniej struktury
    """
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
    """
    Funkcja pobierająca dane i wyświetlająca wykres świecowy
    """
    # URL do API Kucoin, możesz zmienić na inne API
    url = f'https://api.kucoin.com/api/v1/market/candles?symbol={pair}&type={interval}&limit={limit}'
    
    # Pobieranie danych
    response = requests.get(url)
    response_body = response.text
    response_body_json = json.loads(response_body)
    
    # Wyciąganie danych świeczek
    candles_data = response_body_json["data"]
    
    # Formatowanie danych
    formatted_candles_data = format_candles_data(candles_data)
    
    # Przetwarzanie danych przy użyciu pandas
    df = pd.json_normalize(formatted_candles_data)
    df.time = pd.to_datetime(df.time, unit='s')
    df = df.set_index("time")
    
    # Weryfikacja danych
    print("Kolumny:")
    print(df.columns)
    print("\nPodgląd danych:")
    print(df.head())
    
    # Wyświetlanie wykresu
    mpl.plot(
        df,
        type="candle",
        title=f"Wykres świecowy dla {pair}",
        style="yahoo",
        mav=(3, 6, 9)  # Średnie kroczące
    )

# Wywołanie funkcji z domyślnymi parametrami
fetch_and_display_candles()

# Możesz również zmienić parametry wywołania, np.:
# fetch_and_display_candles(pair="ETH-USDT", interval="15m", limit=50)
