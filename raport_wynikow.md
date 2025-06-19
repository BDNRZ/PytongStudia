# Raport: Porównanie Trzech Wariantów Modelu LSTM

## Opis zadania
Bazując na podanym kodzie programu (pliki deep.py, chart.py oraz api.py), zmodyfikowano parametry modelu oraz przygotowano 3 różne wykresy przedstawiające rezultat tych zmian.

## Parametry do modyfikacji

Zgodnie z zadaniem, zmodyfikowano następujące parametry modelu LSTM:

### 1. Liczba jednostek w warstwach LSTM (LSTM units)
- **Wariant 1**: Pierwsza warstwa 100 jednostek, druga warstwa 50 jednostek
- **Wariant 2**: Pierwsza warstwa 200 jednostek, druga warstwa 100 jednostek  
- **Wariant 3**: Pierwsza warstwa 150 jednostek, druga warstwa 75 jednostek

### 2. Stopa dropout (dropout rate)
- **Wariant 1**: Dropout 0.2 (20% jednostek wyłączanych)
- **Wariant 2**: Dropout 0.3 (30% jednostek wyłączanych)
- **Wariant 3**: Dropout 0.4 (40% jednostek wyłączanych)

## Szczegóły implementacji

### Architektura modelu
Każdy wariant składa się z:
- **2 warstw LSTM** (zgodnie z oryginalnym kodem)
- **Warstwy Dropout** po każdej warstwie LSTM
- **Warstwy Dense(1)** jako warstwa wyjściowa
- **Optimizer**: Adam
- **Loss function**: Mean Squared Error

### Parametry treningu (niezmienione)
- **Epoki**: 10
- **Batch size**: 64
- **Validation split**: 0.1
- **Kroki czasowe**: 60
- **Liczba przewidywanych kroków**: 5

## Analiza wpływu parametrów

### Wariant 1: LSTM(100,50), Dropout 0.2
- **Charakterystyka**: Najmniejsza sieć z niskim dropoutem
- **Zalety**: Szybszy trening, mniejsze zużycie pamięci
- **Wady**: Może być podatny na przeuczenie
- **Oczekiwany efekt**: Szybsza zbieżność, ale potencjalnie gorsza generalizacja

### Wariant 2: LSTM(200,100), Dropout 0.3
- **Charakterystyka**: Największa sieć ze średnim dropoutem
- **Zalety**: Większa pojemność modelu, lepsza regularyzacja
- **Wady**: Dłuższy trening, większe zużycie zasobów
- **Oczekiwany efekt**: Lepsze dopasowanie do złożonych wzorców

### Wariant 3: LSTM(150,75), Dropout 0.4
- **Charakterystyka**: Średnia sieć z wysokim dropoutem
- **Zalety**: Silna regularyzacja, stabilność
- **Wady**: Wolniejsza nauka, potencjalnie gorsze dopasowanie
- **Oczekiwany efekt**: Najlepsza generalizacja, ale wolniejsza zbieżność

## Metodologia porównania

### Dane
- **Symbol**: ETHUSDT
- **Interwał**: 15 minut
- **Liczba świeczek**: 1000
- **Podział danych**: 80% treningowe, 20% testowe

### Metryki oceny
- **Mean Squared Error (MSE)** na zbiorze treningowym i testowym
- **R² Score** na zbiorze testowym
- **Wizualna analiza** dopasowania predykcji do rzeczywistych cen
- **Analiza prognoz** na przyszłość (5 kroków)

## Wyniki

### Wykresy
Każdy wariant generuje osobny wykres zawierający:
- **Dane historyczne** (niebieska linia)
- **Predykcje na zbiorze treningowym** (jasnoniebieska linia)
- **Rzeczywiste ceny testowe** (czarna linia)
- **Predykcje na zbiorze testowym** (jasnoniebieska linia)
- **Prognoza na przyszłość** (przerywana linia)

### Pliki wynikowe
- `wariant_1_lstm.png` - LSTM(100,50), Dropout 0.2
- `wariant_2_lstm.png` - LSTM(200,100), Dropout 0.3
- `wariant_3_lstm.png` - LSTM(150,75), Dropout 0.4

## Wnioski

### Wpływ liczby jednostek LSTM
- **Większa liczba jednostek** = większa pojemność modelu, ale dłuższy trening
- **Mniejsza liczba jednostek** = szybszy trening, ale ograniczona pojemność

### Wpływ dropout
- **Niższy dropout** = szybsza nauka, ale ryzyko przeuczenia
- **Wyższy dropout** = lepsza regularyzacja, ale wolniejsza nauka

### Optymalizacja
Najlepszy wariant zależy od:
- **Dostępnych zasobów** (czas, pamięć)
- **Wymaganej dokładności**
- **Czasu odpowiedzi** (dla predykcji w czasie rzeczywistym)

## Implementacja techniczna

### Użyte biblioteki
- **Keras/TensorFlow** - implementacja modeli LSTM
- **scikit-learn** - preprocessing danych (MinMaxScaler)
- **matplotlib** - wizualizacja wyników
- **pandas/numpy** - manipulacja danymi

### Struktura kodu
- **deep.py** - główny skrypt z implementacją trzech wariantów
- **chart.py** - funkcje do wizualizacji
- **api.py** - pobieranie danych z Binance API

## Podsumowanie

Zadanie zostało zrealizowane zgodnie z wymaganiami:
1. ✅ Zmodyfikowano parametry modelu LSTM
2. ✅ Przygotowano 3 różne wykresy
3. ✅ Użyto oryginalnego kodu jako bazy
4. ✅ Zastosowano funkcję plot_predictions z chart.py
5. ✅ Utworzono dokumentację z informacjami o zmienionych parametrach

Wszystkie trzy warianty używają tej samej metody (LSTM), ale z różnymi parametrami architektury, co pozwala na bezpośrednie porównanie wpływu tych parametrów na wyniki predykcji cen kryptowalut. 