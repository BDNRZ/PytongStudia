import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(df, real_prices, predicted_prices, future_predictions, future_steps, interval, candle_count):
    """
    Funkcja do wizualizacji predykcji modelu LSTM
    
    Parametry:
    - df: DataFrame z danymi historycznymi
    - real_prices: rzeczywiste ceny z danych testowych
    - predicted_prices: ceny przewidziane przez model
    - future_predictions: prognozy na przyszłość
    - future_steps: liczba kroków do przewidzenia
    - interval: interwał czasowy
    - candle_count: liczba świeczek
    """
    
    # Konfiguracja stylu wykresu
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Ustawienie tła
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    
    # Indeksy dla danych testowych
    test_start_idx = len(df) - len(real_prices) - future_steps
    test_indices = df.index[test_start_idx:test_start_idx + len(real_prices)]
    
    # Indeksy dla predykcji
    pred_indices = df.index[test_start_idx:test_start_idx + len(predicted_prices)]
    
    # Indeksy dla przyszłych predykcji
    future_indices = pd.date_range(
        start=df.index[-1], 
        periods=future_steps + 1, 
        freq=pd.Timedelta(minutes=int(interval.replace('m', '')))
    )[1:]
    
    # Rysowanie linii
    # Dane historyczne (treningowe)
    ax.plot(df.index[:test_start_idx], df['Close'].iloc[:test_start_idx], 
            color='#4361ee', alpha=0.8, linewidth=1.5, label='Dane historyczne')
    
    # Rzeczywiste ceny (testowe)
    ax.plot(test_indices, real_prices, 
            color='#2d3748', alpha=0.9, linewidth=2, label='Rzeczywiste ceny (test)')
    
    # Predykcje na danych testowych
    ax.plot(pred_indices, predicted_prices, 
            color='#4cc9f0', alpha=0.8, linewidth=2, label='Predykcje (test)')
    
    # Prognozy na przyszłość
    ax.plot(future_indices, future_predictions, 
            color='#00b4d8', linestyle='--', alpha=0.9, linewidth=2.5, label='Prognoza przyszłości')
    
    # Konfiguracja wykresu
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'Predykcja Cen - Model LSTM\n{interval} interwał, {candle_count} świeczek', 
                 pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel('Czas', labelpad=10, fontsize=12)
    ax.set_ylabel('Cena (USDT)', labelpad=10, fontsize=12)
    
    # Legenda
    ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=10)
    
    # Formatowanie osi X
    ax.tick_params(axis='x', rotation=45)
    
    # Dodanie informacji o parametrach modelu
    info_text = f'Kroki czasowe: 60\nEpoki: 10\nBatch size: 64\nDropout: 0.2'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig
