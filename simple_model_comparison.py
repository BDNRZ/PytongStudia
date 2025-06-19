import api
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_simple_predictions(df, time_step, method="moving_average"):
    """
    Funkcja do tworzenia prostych predykcji bez użycia LSTM
    
    Parametry:
    - df: DataFrame z danymi
    - time_step: liczba kroków czasowych
    - method: metoda predykcji ("moving_average", "linear_trend", "exponential_smoothing")
    """
    
    # Przetwarzanie danych
    df_processed = df.copy()
    df_processed['Future_Close'] = df_processed['Close'].shift(-1)
    df_processed.dropna(inplace=True)
    
    # Skalowanie danych
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_processed[['Close', 'Future_Close']])
    
    # Tworzenie danych treningowych i testowych
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    # Tworzenie predykcji na podstawie różnych metod
    predictions = []
    
    for i in range(len(test_data) - time_step):
        if method == "moving_average":
            # Średnia krocząca
            window = test_data[i:i+time_step, 0]
            pred = np.mean(window)
        elif method == "linear_trend":
            # Trend liniowy
            x = np.arange(time_step)
            y = test_data[i:i+time_step, 0]
            coeffs = np.polyfit(x, y, 1)
            pred = coeffs[0] * time_step + coeffs[1]
        elif method == "exponential_smoothing":
            # Wygładzanie wykładnicze
            alpha = 0.3
            window = test_data[i:i+time_step, 0]
            pred = window[0]
            for j in range(1, len(window)):
                pred = alpha * window[j] + (1 - alpha) * pred
        
        predictions.append(pred)
    
    # Konwersja predykcji z powrotem do oryginalnej skali
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_with_zeros = np.hstack((predictions_array, np.zeros((predictions_array.shape[0], 1))))
    predicted_prices = scaler.inverse_transform(predictions_with_zeros)[:, 0]
    
    # Generowanie prognoz na przyszłość
    future_steps = 5
    future_predictions = []
    
    # Ostatnie dane do predykcji przyszłości
    last_data = test_data[-time_step:, 0]
    
    for _ in range(future_steps):
        if method == "moving_average":
            pred = np.mean(last_data)
        elif method == "linear_trend":
            x = np.arange(time_step)
            coeffs = np.polyfit(x, last_data, 1)
            pred = coeffs[0] * time_step + coeffs[1]
        elif method == "exponential_smoothing":
            alpha = 0.3
            pred = last_data[0]
            for j in range(1, len(last_data)):
                pred = alpha * last_data[j] + (1 - alpha) * pred
        
        future_predictions.append(pred)
        # Aktualizacja danych dla następnej predykcji
        last_data = np.roll(last_data, -1)
        last_data[-1] = pred
    
    # Konwersja przyszłych predykcji
    future_prices_array = np.array(future_predictions).reshape(-1, 1)
    future_prices_scaled = np.hstack((future_prices_array, np.zeros((future_prices_array.shape[0], 1))))
    future_predictions = scaler.inverse_transform(future_prices_scaled)[:, 0]
    
    # Przygotowanie rzeczywistych cen
    real_prices = df_processed['Close'].iloc[train_size + time_step:].values
    
    return df_processed, real_prices, predicted_prices, future_predictions, method

def plot_model_comparison(results, interval, candle_count):
    """Funkcja do tworzenia porównania trzech modeli"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle('Porównanie Modeli Predykcji z Różnymi Parametrami', fontsize=16, fontweight='bold')
    
    for i, (df, real_prices, predicted_prices, future_predictions, method) in enumerate(results):
        ax = axes[i]
        
        # Konfiguracja stylu
        ax.set_facecolor('#f8f9fa')
        
        # Indeksy
        train_size = int(len(df) * 0.8)
        test_start_idx = train_size
        test_indices = df.index[test_start_idx:test_start_idx + len(real_prices)]
        pred_indices = df.index[test_start_idx:test_start_idx + len(predicted_prices)]
        
        # Indeksy dla przyszłych predykcji
        future_indices = pd.date_range(
            start=df.index[-1], 
            periods=6, 
            freq=pd.Timedelta(minutes=int(interval.replace('m', '')))
        )[1:]
        
        # Rysowanie linii
        ax.plot(df.index[:test_start_idx], df['Close'].iloc[:test_start_idx], 
                color='#4361ee', alpha=0.8, linewidth=1.5, label='Dane historyczne')
        ax.plot(test_indices, real_prices, 
                color='#2d3748', alpha=0.9, linewidth=2, label='Rzeczywiste ceny')
        ax.plot(pred_indices, predicted_prices, 
                color='#4cc9f0', alpha=0.8, linewidth=2, label='Predykcje')
        ax.plot(future_indices, future_predictions, 
                color='#00b4d8', linestyle='--', alpha=0.9, linewidth=2.5, label='Prognoza')
        
        # Konfiguracja wykresu
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'Model {i+1}: {method}', pad=10, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cena (USDT)', fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        
        # Dodanie informacji o parametrach
        if i == 0:
            info_text = f'Metoda: Średnia krocząca\nKroki czasowe: 60\nParametr: Okno 60'
        elif i == 1:
            info_text = f'Metoda: Trend liniowy\nKroki czasowe: 100\nParametr: Regresja liniowa'
        else:
            info_text = f'Metoda: Wygładzanie wykładnicze\nKroki czasowe: 60\nParametr: α = 0.3'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Czas', fontsize=10)
    plt.tight_layout()
    return fig

# Główna funkcja
def main():
    # Pobieranie danych
    symbol = 'ETHUSDT'
    interval = '15m'
    candle_count = 1000
    df = api.fetch_data(symbol, interval, candle_count)
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Open_time', inplace=True)
    
    print("Pobieranie danych zakończone. Rozpoczynam analizę modeli...")
    
    # Model 1: Średnia krocząca z większym oknem
    print("Analiza Modelu 1: Średnia krocząca...")
    model1_results = create_simple_predictions(
        df=df,
        time_step=60,
        method="moving_average"
    )
    
    # Model 2: Trend liniowy z większymi krokami czasowymi
    print("Analiza Modelu 2: Trend liniowy...")
    model2_results = create_simple_predictions(
        df=df,
        time_step=100,  # Zwiększone z 60
        method="linear_trend"
    )
    
    # Model 3: Wygładzanie wykładnicze z różnymi parametrami
    print("Analiza Modelu 3: Wygładzanie wykładnicze...")
    model3_results = create_simple_predictions(
        df=df,
        time_step=60,
        method="exponential_smoothing"
    )
    
    # Tworzenie wykresu porównawczego
    print("Tworzenie wykresu porównawczego...")
    results = [model1_results, model2_results, model3_results]
    comparison_fig = plot_model_comparison(results, interval, candle_count)
    
    # Zapisanie wykresu
    comparison_fig.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Wykres porównawczy zapisany jako 'model_comparison.png'")
    
    # Wyświetlenie wyników
    print("\n=== PODSUMOWANIE WYNIKÓW ===")
    for i, (df, real_prices, predicted_prices, future_predictions, method) in enumerate(results, 1):
        # Obliczenie błędu średniokwadratowego
        mse = np.mean((real_prices - predicted_prices) ** 2)
        mae = np.mean(np.abs(real_prices - predicted_prices))
        
        print(f"\nModel {i}: {method}")
        print(f"  MSE (Mean Squared Error): {mse:.4f}")
        print(f"  MAE (Mean Absolute Error): {mae:.4f}")
        print(f"  RMSE (Root Mean Squared Error): {np.sqrt(mse):.4f}")
    
    plt.show()

if __name__ == "__main__":
    main() 