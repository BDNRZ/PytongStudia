import api
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from chart import plot_predictions

def create_dataset(data, time_step):
    """Funkcja tworząca dane do modelu LSTM"""
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 1])
    return np.array(X), np.array(y)

def train_and_predict_model(df, time_step, lstm_units_1, lstm_units_2, lstm_units_3=None, 
                          epochs=10, batch_size=64, dropout_rate=0.2, 
                          learning_rate=0.001, dense_units=None, model_name="Model"):
    """
    Funkcja do trenowania modelu i generowania predykcji
    
    Parametry:
    - df: DataFrame z danymi
    - time_step: liczba kroków czasowych
    - lstm_units_1: liczba jednostek w pierwszej warstwie LSTM
    - lstm_units_2: liczba jednostek w drugiej warstwie LSTM
    - lstm_units_3: liczba jednostek w trzeciej warstwie LSTM (opcjonalnie)
    - epochs: liczba epok
    - batch_size: rozmiar partii
    - dropout_rate: stopa dropout
    - learning_rate: współczynnik uczenia
    - dense_units: liczba jednostek w warstwie Dense (opcjonalnie)
    - model_name: nazwa modelu
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
    
    # Tworzenie zestawów danych
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Reshape input do formatu [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Budowanie modelu LSTM
    model = Sequential()
    
    # Pierwsza warstwa LSTM
    model.add(LSTM(units=lstm_units_1, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(dropout_rate))
    
    # Druga warstwa LSTM
    model.add(LSTM(units=lstm_units_2, return_sequences=(lstm_units_3 is not None)))
    model.add(Dropout(dropout_rate))
    
    # Trzecia warstwa LSTM (jeśli podana)
    if lstm_units_3 is not None:
        model.add(LSTM(units=lstm_units_3))
        model.add(Dropout(dropout_rate))
    
    # Warstwa Dense (jeśli podana)
    if dense_units is not None:
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Warstwa wyjściowa
    model.add(Dense(1))
    
    # Kompilacja modelu
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Trening modelu
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                       validation_data=(X_test, y_test), verbose=0)
    
    # Predykcja
    predictions = model.predict(X_test)
    predictions_with_zeros = np.hstack((predictions, np.zeros((predictions.shape[0], 1))))
    predicted_prices = scaler.inverse_transform(predictions_with_zeros)[:, 0]
    
    # Generowanie prognoz na przyszłość
    future_steps = 5
    future_predictions = []
    last_sequence = X_test[-1:]
    
    for _ in range(future_steps):
        last_prediction = model.predict(last_sequence)
        future_predictions.append(last_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = last_prediction
    
    future_prices_array = np.array(future_predictions).reshape(-1, 1)
    future_prices_scaled = np.hstack((future_prices_array, np.zeros((future_prices_array.shape[0], 1))))
    future_predictions = scaler.inverse_transform(future_prices_scaled)[:, 0]
    
    # Przygotowanie rzeczywistych cen
    test_data_with_zeros = np.hstack((
        (test_data[time_step+1:, 0].reshape(-1, 1)),
        np.zeros((test_data[time_step+1:, 0].reshape(-1, 1).shape[0], 1))
    ))
    real_prices_scaled = scaler.inverse_transform(test_data_with_zeros)[:, 0]
    
    return df_processed, real_prices_scaled, predicted_prices, future_predictions, history, model_name

def plot_model_comparison(results, interval, candle_count):
    """Funkcja do tworzenia porównania trzech modeli"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle('Porównanie Modeli LSTM z Różnymi Parametrami', fontsize=16, fontweight='bold')
    
    for i, (df, real_prices, predicted_prices, future_predictions, history, model_name) in enumerate(results):
        ax = axes[i]
        
        # Konfiguracja stylu
        ax.set_facecolor('#f8f9fa')
        
        # Indeksy
        test_start_idx = len(df) - len(real_prices) - 5
        test_indices = df.index[test_start_idx:test_start_idx + len(real_prices)]
        pred_indices = df.index[test_start_idx:test_start_idx + len(predicted_prices)]
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
        ax.set_title(f'{model_name}', pad=10, fontsize=12, fontweight='bold')
        ax.set_ylabel('Cena (USDT)', fontsize=10)
        ax.legend(frameon=True, facecolor='white', framealpha=1, fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        
        # Dodanie informacji o loss
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        ax.text(0.02, 0.98, f'Loss: {final_loss:.4f}\nVal Loss: {final_val_loss:.4f}', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
    
    print("Pobieranie danych zakończone. Rozpoczynam trenowanie modeli...")
    
    # Model 1: Zwiększona liczba jednostek LSTM i epok
    print("Trenowanie Modelu 1: Zwiększone jednostki LSTM i epoki...")
    model1_results = train_and_predict_model(
        df=df,
        time_step=60,
        lstm_units_1=200,  # Zwiększone z 100
        lstm_units_2=150,  # Zwiększone z 75
        epochs=20,         # Zwiększone z 10
        batch_size=64,
        dropout_rate=0.2,
        learning_rate=0.001,
        model_name="Model 1: Większe jednostki LSTM (200, 150) + więcej epok (20)"
    )
    
    # Model 2: Dodana trzecia warstwa LSTM i większy dropout
    print("Trenowanie Modelu 2: Trzecia warstwa LSTM i większy dropout...")
    model2_results = train_and_predict_model(
        df=df,
        time_step=60,
        lstm_units_1=100,
        lstm_units_2=75,
        lstm_units_3=50,   # Dodana trzecia warstwa
        epochs=10,
        batch_size=32,     # Zmniejszone z 64
        dropout_rate=0.4,  # Zwiększone z 0.2
        learning_rate=0.001,
        model_name="Model 2: Trzecia warstwa LSTM (50) + większy dropout (0.4) + mniejszy batch (32)"
    )
    
    # Model 3: Większe kroki czasowe i dodatkowa warstwa Dense
    print("Trenowanie Modelu 3: Większe kroki czasowe i dodatkowa warstwa Dense...")
    model3_results = train_and_predict_model(
        df=df,
        time_step=100,     # Zwiększone z 60
        lstm_units_1=100,
        lstm_units_2=75,
        epochs=10,
        batch_size=64,
        dropout_rate=0.2,
        learning_rate=0.01,  # Zwiększone z 0.001
        dense_units=50,      # Dodana warstwa Dense
        model_name="Model 3: Większe kroki czasowe (100) + warstwa Dense (50) + większy learning rate (0.01)"
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
    for i, (df, real_prices, predicted_prices, future_predictions, history, model_name) in enumerate(results, 1):
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"\nModel {i}: {model_name}")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Różnica (Loss - Val Loss): {final_loss - final_val_loss:.4f}")
    
    plt.show()

if __name__ == "__main__":
    main() 