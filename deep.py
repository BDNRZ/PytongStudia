import api # Importowanie modułu api.py
import numpy as np # Importowanie modułu numpy
import pandas as pd # Importowanie modułu pandas
from sklearn.preprocessing import MinMaxScaler # Importowanie klasy MinMaxScaler z modułu sklearn.preprocessing
from sklearn.neural_network import MLPRegressor # Importowanie MLPRegressor zamiast Keras
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Używamy backendu Agg aby nie wyświetlać wykresów
import matplotlib.pyplot as plt

def create_lstm_model_variant(variant_number, input_size):
    """Tworzy model sieci neuronowej z określonymi parametrami dla danego wariantu"""
    
    if variant_number == 1:
        # Wariant 1: 2 warstwy ukryte, mniejsze jednostki, niski dropout
        print("Tworzenie Wariantu 1: 2 warstwy ukryte (100, 50), regularyzacja 0.2")
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.2,  # Regularyzacja L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
    elif variant_number == 2:
        # Wariant 2: 2 warstwy ukryte, większe jednostki, średni dropout
        print("Tworzenie Wariantu 2: 2 warstwy ukryte (200, 100), regularyzacja 0.3")
        model = MLPRegressor(
            hidden_layer_sizes=(200, 100),
            activation='relu',
            solver='adam',
            alpha=0.3,  # Regularyzacja L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
    elif variant_number == 3:
        # Wariant 3: 2 warstwy ukryte, średnie jednostki, wysoki dropout
        print("Tworzenie Wariantu 3: 2 warstwy ukryte (150, 75), regularyzacja 0.4")
        model = MLPRegressor(
            hidden_layer_sizes=(150, 75),
            activation='relu',
            solver='adam',
            alpha=0.4,  # Regularyzacja L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    
    return model

def prepare_data(df, time_step):
    """Przygotowuje dane do treningu"""
    df['Future_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close', 'Future_Close']])
    
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    X_train, y_train = [], []
    for i in range(len(train_data) - time_step):
        X_train.append(train_data[i:(i + time_step), 0])
        y_train.append(train_data[i + time_step, 1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    X_test, y_test = [], []
    for i in range(len(test_data) - time_step):
        X_test.append(test_data[i:(i + time_step), 0])
        y_test.append(test_data[i + time_step, 1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    return X_train, y_train, X_test, y_test, scaler, train_size

def train_and_predict_model(variant_number, X_train, y_train, X_test, y_test, scaler, time_step):
    """Trenuje model i tworzy predykcje"""
    print(f"\n{'='*60}")
    print(f"TRENOWANIE WARIANTU {variant_number}")
    print(f"{'='*60}")
    
    try:
        # Tworzenie modelu
        model = create_lstm_model_variant(variant_number, X_train.shape[1])
        
        # Trening modelu
        print(f"Rozpoczynam trening...")
        model.fit(X_train, y_train)
        
        # Predykcje
        print(f"Tworzenie predykcji...")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Odwrócenie skalowania
        train_pred_reshaped = train_pred.reshape(-1, 1)
        test_pred_reshaped = test_pred.reshape(-1, 1)
        
        train_pred_with_zeros = np.hstack((train_pred_reshaped, np.zeros((train_pred_reshaped.shape[0], 1))))
        test_pred_with_zeros = np.hstack((test_pred_reshaped, np.zeros((test_pred_reshaped.shape[0], 1))))
        
        train_pred_original = scaler.inverse_transform(train_pred_with_zeros)[:, 0]
        test_pred_original = scaler.inverse_transform(test_pred_with_zeros)[:, 0]
        
        # Generowanie prognoz przyszłości
        future_steps = 5
        future_predictions = []
        last_sequence = X_test[-1:].flatten()
        
        for _ in range(future_steps):
            next_pred = model.predict(last_sequence.reshape(1, -1))
            future_predictions.append(next_pred[0])
            
            # Aktualizacja sekwencji
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred[0]
        
        future_prices_array = np.array(future_predictions).reshape(-1, 1)
        future_prices_scaled = np.hstack((future_prices_array, np.zeros((future_prices_array.shape[0], 1))))
        future_predictions_original = scaler.inverse_transform(future_prices_scaled)[:, 0]
        
        # Przygotowanie rzeczywistych cen
        y_test_reshaped = y_test.reshape(-1, 1)
        y_test_with_zeros = np.hstack((y_test_reshaped, np.zeros((y_test_reshaped.shape[0], 1))))
        real_prices_original = scaler.inverse_transform(y_test_with_zeros)[:, 0]
        
        # Obliczanie metryk
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Wyniki dla Wariantu {variant_number}:")
        print(f"Train MSE: {train_mse:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Liczba iteracji: {model.n_iter_}")
        print(f"Przyszłe predykcje: {future_predictions_original}")
        
        return test_pred_original, real_prices_original, future_predictions_original
        
    except Exception as e:
        print(f"Błąd w Wariantu {variant_number}: {e}")
        return None, None, None

def create_plot(df, pred_prices, real_prices, future_pred, variant_num, train_size, time_step):
    """Tworzy wykres dla danego wariantu"""
    try:
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Ustawienie stylu
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#ffffff')
        
        # Rysowanie danych historycznych (treningowych)
        ax.plot(df.index[:train_size], df['Close'].iloc[:train_size],
                color='#4361ee', alpha=0.8, linewidth=1.5, label='Dane historyczne')
        
        # Rysowanie rzeczywistych cen (testowych)
        test_start = train_size + time_step
        test_end = test_start + len(real_prices)
        test_indices = df.index[test_start:test_end]
        ax.plot(test_indices, real_prices,
                color='#2d3748', alpha=0.9, linewidth=2, label='Rzeczywiste ceny (test)')
        
        # Rysowanie predykcji testowych
        pred_indices = df.index[test_start:test_start+len(pred_prices)]
        ax.plot(pred_indices, pred_prices,
                color='#4cc9f0', alpha=0.8, linewidth=2, label='Predykcje (test)')
        
        # Rysowanie przyszłych predykcji
        future_index = pd.date_range(start=df.index[-1], periods=6, freq='15min')[1:]
        ax.plot(future_index, future_pred,
                color='#00b4d8', linestyle='--', alpha=0.9, linewidth=2.5, label='Prognoza')
        
        # Konfiguracja wykresu
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tytuł z parametrami
        if variant_num == 1:
            title = "Wariant 1: Warstwy(100,50), Regularyzacja 0.2"
        elif variant_num == 2:
            title = "Wariant 2: Warstwy(200,100), Regularyzacja 0.3"
        else:
            title = "Wariant 3: Warstwy(150,75), Regularyzacja 0.4"
        
        ax.set_title(f'Predykcja Cen - Model Sieci Neuronowej - {title}', 
                     pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel('Czas', labelpad=10, fontsize=12)
        ax.set_ylabel('Cena (USDT)', labelpad=10, fontsize=12)
        ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Błąd podczas tworzenia wykresu dla Wariantu {variant_num}: {e}")
        return None

def main():
    # Parametry
    symbol = 'ETHUSDT'
    interval = '15m'
    candle_count = 1000
    time_step = 60
    
    print(f"Pobieranie danych dla {symbol}, interwał: {interval}, liczba świeczek: {candle_count}")
    
    try:
        # Pobieranie danych
        df = api.fetch_data(symbol, interval, candle_count)
        df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
        df.set_index('Open_time', inplace=True)
        
        # Przygotowanie danych
        X_train, y_train, X_test, y_test, scaler, train_size = prepare_data(df, time_step)
        
        print(f"\nDane przygotowane:")
        print(f"Rozmiar zbioru treningowego: {X_train.shape}")
        print(f"Rozmiar zbioru testowego: {X_test.shape}")
        print(f"Liczba kroków czasowych: {time_step}")
        
        # Lista do przechowywania wyników
        all_predictions = []
        all_real_prices = []
        all_future_predictions = []
        
        # Trenowanie wszystkich trzech wariantów
        for variant in [1, 2, 3]:
            pred_prices, real_prices, future_pred = train_and_predict_model(
                variant, X_train, y_train, X_test, y_test, scaler, time_step
            )
            
            if pred_prices is not None:
                all_predictions.append(pred_prices)
                all_real_prices.append(real_prices)
                all_future_predictions.append(future_pred)
        
        # Tworzenie wykresów dla każdego wariantu
        print(f"\nTworzenie wykresów...")
        
        for i, (pred_prices, real_prices, future_pred) in enumerate(zip(all_predictions, all_real_prices, all_future_predictions)):
            variant_num = i + 1
            print(f"Tworzenie wykresu dla Wariantu {variant_num}...")
            
            # Tworzenie wykresu
            fig = create_plot(df, pred_prices, real_prices, future_pred, variant_num, train_size, time_step)
            
            if fig is not None:
                # Zapisywanie wykresu
                filename = f'wariant_{variant_num}_neural_network.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Wykres zapisany jako '{filename}'")
                plt.close()  # Zamykamy wykres aby zwolnić pamięć
        
        print(f"\nWszystkie warianty zakończone!")
        print(f"Utworzone pliki:")
        print(f"- wariant_1_neural_network.png - Warstwy(100,50), Regularyzacja 0.2")
        print(f"- wariant_2_neural_network.png - Warstwy(200,100), Regularyzacja 0.3") 
        print(f"- wariant_3_neural_network.png - Warstwy(150,75), Regularyzacja 0.4")
        
    except Exception as e:
        print(f"Błąd główny: {e}")

if __name__ == "__main__":
    main()