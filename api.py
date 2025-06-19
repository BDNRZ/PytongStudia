import requests
import pandas as pd

def fetch_data(symbol, interval, candle_count):
    url = 'https://api.binance.com/api/v3/klines'
    limit = 1000  # Maksymalna liczba świeczek na zapytanie
    all_candles = []

    if candle_count <= limit:
        # Jeśli potrzebujemy mniej lub równo 1000 świeczek
        params = {'symbol': symbol, 'interval': interval, 'limit': candle_count}
        data = requests.get(url, params=params).json()

        # add data to all_candles at the beginning
        all_candles = data + all_candles
    else:
        # Jeśli potrzebujemy więcej niż 1000 świeczek
        iterations = candle_count // limit  # Liczba pełnych iteracji
        remaining = candle_count % limit  # Pozostałe świeczki do pobrania

        # Pobieranie danych w partiach
        end_time = None  # Początkowo brak konkretnego czasu zakończenia
        for _ in range(iterations):
            print(end_time)
            params = {'symbol': symbol, 'interval': interval, 'limit': limit, 'endTime': end_time}
            data = requests.get(url, params=params).json()
            all_candles = data + all_candles
            end_time = data[0][0]  # Ustawienie czasu zakończenia na początek pobranej partii

        # Pobieranie pozostałych świeczek
        if remaining > 0:
            params = {'symbol': symbol, 'interval': interval, 'limit': remaining, 'endTime': end_time}
            data = requests.get(url, params=params).json()
            all_candles = data + all_candles

    # Tworzenie DataFrame z pobranych danych
    df = pd.DataFrame(all_candles, columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
    df['Close'] = df['Close'].astype(float)

    return df