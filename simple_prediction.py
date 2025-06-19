import api
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def prepare_data(df, window_size=60):
    """Przygotowuje dane do analizy"""
    print("\nPrzygotowywanie danych...")
    df_processed = df.copy()
    df_processed['Close'] = pd.to_numeric(df_processed['Close'])
    
    # Tworzenie cech na podstawie poprzednich wartości
    print(f"Tworzenie {window_size} opóźnionych cech...")
    for i in range(1, window_size + 1):
        df_processed[f'Close_lag_{i}'] = df_processed['Close'].shift(i)
    
    # Dodanie dodatkowych cech
    df_processed['MA_5'] = df_processed['Close'].rolling(window=5).mean()
    df_processed['MA_10'] = df_processed['Close'].rolling(window=10).mean()
    df_processed['MA_20'] = df_processed['Close'].rolling(window=20).mean()
    
    # Usuwanie wierszy z brakującymi danymi
    df_processed.dropna(inplace=True)
    print(f"Liczba próbek po przygotowaniu: {len(df_processed)}")
    
    return df_processed

def create_neural_network_variant(hidden_layer_sizes, dropout_rate, **kwargs):
    """Tworzy sieć neuronową z określonymi parametrami"""
    # MLPRegressor nie ma bezpośredniego parametru dropout, ale możemy symulować regularyzację
    # używając alpha (L2 regularization) i early_stopping
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=dropout_rate * 0.1,  # Symulacja dropout przez regularyzację
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        **kwargs
    )

def train_and_predict(model, X_train, y_train, X_test, y_test, model_name):
    """Trenuje model i tworzy predykcje"""
    print(f"\nTrenowanie modelu: {model_name}")
    print(f"Rozmiar zbioru treningowego: {X_train.shape}")
    print(f"Rozmiar zbioru testowego: {X_test.shape}")
    
    # Trenowanie modelu
    model.fit(X_train, y_train)
    
    # Predykcje
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Obliczanie metryk
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"Wyniki dla {model_name}:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Liczba iteracji: {model.n_iter_}")
    
    return test_pred, train_pred

def create_future_predictions(model, last_known_values, steps=5):
    """Tworzy predykcje na przyszłość"""
    future_predictions = []
    current_values = last_known_values.copy()
    
    for _ in range(steps):
        next_pred = model.predict(current_values.reshape(1, -1))
        future_predictions.append(next_pred[0])
        
        # Aktualizacja wartości dla następnej predykcji
        current_values = np.roll(current_values, 1)
        current_values[0] = next_pred[0]
    
    return np.array(future_predictions)

def plot_predictions(df, predictions_list, train_predictions_list, future_predictions_list, model_names, train_size):
    """Tworzy wykres porównawczy dla trzech wariantów sieci neuronowej"""
    print("\nTworzenie wykresu...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle('Porównanie Trzech Wariantów Sieci Neuronowej', fontsize=16, fontweight='bold')
    
    for i, (predictions, train_pred, future_pred, model_name) in enumerate(zip(predictions_list, train_predictions_list, future_predictions_list, model_names)):
        ax = axes[i]
        ax.set_facecolor('#f8f9fa')
        
        # Rysowanie danych historycznych (treningowych)
        ax.plot(df.index[:train_size], df['Close'].iloc[:train_size],
                color='#4361ee', alpha=0.8, linewidth=1.5, label='Dane historyczne')
        
        # Rysowanie predykcji treningowych
        train_indices = df.index[:len(train_pred)]
        ax.plot(train_indices, train_pred,
                color='#4cc9f0', alpha=0.6, linewidth=1.5, label='Predykcje (trening)')
        
        # Rysowanie rzeczywistych cen (testowych)
        ax.plot(df.index[train_size:], df['Close'].iloc[train_size:],
                color='#2d3748', alpha=0.9, linewidth=2, label='Rzeczywiste ceny (test)')
        
        # Rysowanie predykcji testowych
        test_indices = df.index[train_size:train_size+len(predictions)]
        ax.plot(test_indices, predictions,
                color='#4cc9f0', alpha=0.8, linewidth=2, label='Predykcje (test)')
        
        # Rysowanie przyszłych predykcji
        future_index = pd.date_range(start=df.index[-1], periods=6, freq='15min')[1:]
        ax.plot(future_index, future_pred,
                color='#00b4d8', linestyle='--', alpha=0.9, linewidth=2.5, label='Prognoza')
        
        # Konfiguracja wykresu
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'Wariant {i+1}: {model_name}', pad=10, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cena (USDT)', fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=9)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    # Parametry
    symbol = 'ETHUSDT'
    interval = '15m'
    candle_count = 1000
    window_size = 60
    
    print(f"Pobieranie danych dla {symbol}, interwał: {interval}, liczba świeczek: {candle_count}")
    
    # Pobieranie i przygotowanie danych
    df = api.fetch_data(symbol, interval, candle_count)
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Open_time', inplace=True)
    df = prepare_data(df, window_size)
    
    # Określenie rozmiaru zbioru treningowego
    train_size = int(len(df) * 0.8)
    print(f"\nPodział danych - zbiór treningowy: {train_size} próbek, testowy: {len(df) - train_size} próbek")
    
    # Przygotowanie cech
    feature_columns = [f'Close_lag_{i}' for i in range(1, window_size + 1)] + ['MA_5', 'MA_10', 'MA_20']
    X = df[feature_columns]
    y = df['Close']
    
    # Podział na zbiór treningowy i testowy
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Parametry trzech wariantów sieci neuronowej (jak w raporcie)
    model_variants = [
        ((100, 50), 0.2, "Sieć neuronowa: warstwa 1 (100 jednostek), warstwa 2 (50 jednostek), dropout 0.2"),
        ((200, 100), 0.3, "Sieć neuronowa: warstwa 1 (200 jednostek), warstwa 2 (100 jednostek), dropout 0.3"),
        ((150, 75), 0.4, "Sieć neuronowa: warstwa 1 (150 jednostek), warstwa 2 (75 jednostek), dropout 0.4")
    ]
    
    predictions_list = []
    train_predictions_list = []
    future_predictions_list = []
    model_names = []
    
    print("\nRozpoczynam trenowanie trzech wariantów sieci neuronowej...")
    
    # Trenowanie i predykcje dla każdego wariantu
    for hidden_layers, dropout_rate, name in model_variants:
        print(f"\n{'='*50}")
        print(f"Wariant: {name}")
        print(f"Parametry: Warstwy ukryte {hidden_layers}, Dropout {dropout_rate}")
        print(f"{'='*50}")
        
        # Tworzenie modelu
        model = create_neural_network_variant(hidden_layers, dropout_rate)
        
        # Trenowanie i predykcje
        test_pred, train_pred = train_and_predict(model, X_train, y_train, X_test, y_test, name)
        
        # Predykcje przyszłości
        future_pred = create_future_predictions(model, X.iloc[-1].values)
        
        predictions_list.append(test_pred)
        train_predictions_list.append(train_pred)
        future_predictions_list.append(future_pred)
        model_names.append(name)
        
        print(f"Przyszłe predykcje dla {name}:")
        for i, pred in enumerate(future_pred, 1):
            print(f"  Krok {i}: {pred:.2f}")
    
    # Tworzenie wykresu
    fig = plot_predictions(df, predictions_list, train_predictions_list, future_predictions_list, model_names, train_size)
    
    # Zapisywanie wykresu
    print("\nZapisywanie wykresu...")
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Wykres zapisany jako 'model_comparison.png'")
    
    # Wyświetlanie wykresu
    plt.show()
    print("\nAnaliza zakończona!")

if __name__ == "__main__":
    main() 