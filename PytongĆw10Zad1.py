# Analiza danych sprzedażowych sklepu "ElectroGadżet" - Q1 2025
# Google Colab Script

# ============================================================================
# 1. IMPORT BIBLIOTEK I KONFIGURACJA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja stylu wykresów
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 2. WCZYTANIE I CZYSZCZENIE DANYCH
# ============================================================================

print("=== WCZYTANIE I CZYSZCZENIE DANYCH ===")

# Wczytanie danych
df = pd.read_csv('data.csv')

print(f"Początkowa liczba wierszy: {len(df)}")
print(f"Początkowa liczba kolumn: {len(df.columns)}")
print("\nPierwsze 5 wierszy:")
print(df.head())

print("\nInformacje o typach danych:")
print(df.info())

print("\nStatystyki opisowe:")
print(df.describe())

# ============================================================================
# 3. CZYSZCZENIE DANYCH
# ============================================================================

print("\n=== CZYSZCZENIE DANYCH ===")

# Sprawdzenie brakujących wartości
print("Brakujące wartości:")
print(df.isnull().sum())

# Usunięcie wierszy z brakującymi danymi
df_clean = df.dropna()
print(f"\nLiczba wierszy po usunięciu brakujących danych: {len(df_clean)}")

# Czyszczenie kolumny Cena_Jednostkowa
def clean_price(price):
    if pd.isna(price):
        return np.nan
    if isinstance(price, str):
        # Usunięcie "PLN" i spacji
        price = price.replace('PLN', '').replace(' ', '').strip()
        # Usunięcie cudzysłowów
        price = price.replace('"', '').replace('"', '')
    try:
        return float(price)
    except:
        return np.nan

df_clean['Cena_Jednostkowa'] = df_clean['Cena_Jednostkowa'].apply(clean_price)

# Usunięcie wierszy z nieprawidłowymi cenami
df_clean = df_clean.dropna(subset=['Cena_Jednostkowa'])
print(f"Liczba wierszy po czyszczeniu cen: {len(df_clean)}")

# Czyszczenie kolumny Ilosc
df_clean['Ilosc'] = pd.to_numeric(df_clean['Ilosc'], errors='coerce')
df_clean = df_clean.dropna(subset=['Ilosc'])
print(f"Liczba wierszy po czyszczeniu ilości: {len(df_clean)}")

# Czyszczenie kolumny Data_Transakcji
df_clean['Data_Transakcji'] = pd.to_datetime(df_clean['Data_Transakcji'], errors='coerce')
df_clean = df_clean.dropna(subset=['Data_Transakcji'])
print(f"Liczba wierszy po czyszczeniu dat: {len(df_clean)}")

# Czyszczenie kolumny Miasto
df_clean['Miasto'] = df_clean['Miasto'].str.strip().str.title()
print(f"Liczba wierszy po czyszczeniu miast: {len(df_clean)}")

# Czyszczenie kolumny Kategoria
df_clean['Kategoria'] = df_clean['Kategoria'].str.strip().str.title()
print(f"Liczba wierszy po czyszczeniu kategorii: {len(df_clean)}")

# Dodanie kolumny z wartością transakcji
df_clean['Wartosc_Transakcji'] = df_clean['Ilosc'] * df_clean['Cena_Jednostkowa']

# Dodanie kolumn z datą
df_clean['Data'] = df_clean['Data_Transakcji'].dt.date
df_clean['Miesiac'] = df_clean['Data_Transakcji'].dt.month
df_clean['Dzien'] = df_clean['Data_Transakcji'].dt.day

print(f"\nKońcowa liczba wierszy: {len(df_clean)}")

# ============================================================================
# 4. ANALIZA OGÓLNA
# ============================================================================

print("\n=== ANALIZA OGÓLNA ===")

# Podstawowe statystyki
total_transactions = len(df_clean)
total_revenue = df_clean['Wartosc_Transakcji'].sum()
avg_transaction_value = df_clean['Wartosc_Transakcji'].mean()
unique_customers = df_clean['ID_Klienta'].nunique()
unique_products = df_clean['Produkt'].nunique()

print(f"Łączna liczba transakcji: {total_transactions}")
print(f"Łączny przychód: {total_revenue:,.2f} PLN")
print(f"Średnia wartość transakcji: {avg_transaction_value:,.2f} PLN")
print(f"Liczba unikalnych klientów: {unique_customers}")
print(f"Liczba unikalnych produktów: {unique_products}")

# ============================================================================
# 5. ANALIZA TRENDÓW CZASOWYCH
# ============================================================================

print("\n=== ANALIZA TRENDÓW CZASOWYCH ===")

# Transakcje według miesięcy
monthly_transactions = df_clean.groupby('Miesiac').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum'
}).rename(columns={'ID_Transakcji': 'Liczba_Transakcji', 'Wartosc_Transakcji': 'Przychod'})

print("Transakcje według miesięcy:")
print(monthly_transactions)

# Wykres transakcji według miesięcy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Liczba transakcji
monthly_transactions['Liczba_Transakcji'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Liczba Transakcji według Miesięcy')
ax1.set_xlabel('Miesiąc')
ax1.set_ylabel('Liczba Transakcji')
ax1.tick_params(axis='x', rotation=0)

# Przychód
monthly_transactions['Przychod'].plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Przychód według Miesięcy')
ax2.set_xlabel('Miesiąc')
ax2.set_ylabel('Przychód (PLN)')
ax2.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# ============================================================================
# 6. ANALIZA KATEGORII PRODUKTÓW
# ============================================================================

print("\n=== ANALIZA KATEGORII PRODUKTÓW ===")

# Analiza kategorii
category_analysis = df_clean.groupby('Kategoria').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum',
    'Ilosc': 'sum'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Przychod',
    'Ilosc': 'Liczba_Sztuk'
}).sort_values('Przychod', ascending=False)

print("Analiza kategorii produktów:")
print(category_analysis)

# Wykres kategorii
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Liczba transakcji według kategorii
category_analysis['Liczba_Transakcji'].plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Liczba Transakcji według Kategorii')
ax1.set_xlabel('Kategoria')
ax1.set_ylabel('Liczba Transakcji')
ax1.tick_params(axis='x', rotation=45)

# Przychód według kategorii
category_analysis['Przychod'].plot(kind='bar', ax=ax2, color='gold')
ax2.set_title('Przychód według Kategorii')
ax2.set_xlabel('Kategoria')
ax2.set_ylabel('Przychód (PLN)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 7. ANALIZA NAJPOPULARNIEJSZYCH PRODUKTÓW
# ============================================================================

print("\n=== ANALIZA NAJPOPULARNIEJSZYCH PRODUKTÓW ===")

# Top 10 produktów według liczby transakcji
top_products_by_transactions = df_clean.groupby('Produkt').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Przychod'
}).sort_values('Liczba_Transakcji', ascending=False).head(10)

print("Top 10 produktów według liczby transakcji:")
print(top_products_by_transactions)

# Top 10 produktów według przychodu
top_products_by_revenue = df_clean.groupby('Produkt').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Przychod'
}).sort_values('Przychod', ascending=False).head(10)

print("\nTop 10 produktów według przychodu:")
print(top_products_by_revenue)

# Wykres top produktów
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Top 5 według liczby transakcji
top_products_by_transactions.head(5)['Liczba_Transakcji'].plot(kind='barh', ax=ax1, color='lightblue')
ax1.set_title('Top 5 Produktów według Liczby Transakcji')
ax1.set_xlabel('Liczba Transakcji')

# Top 5 według przychodu
top_products_by_revenue.head(5)['Przychod'].plot(kind='barh', ax=ax2, color='lightgreen')
ax2.set_title('Top 5 Produktów według Przychodu')
ax2.set_xlabel('Przychód (PLN)')

plt.tight_layout()
plt.show()

# ============================================================================
# 8. ANALIZA MIAST
# ============================================================================

print("\n=== ANALIZA MIAST ===")

# Analiza miast
city_analysis = df_clean.groupby('Miasto').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum',
    'ID_Klienta': 'nunique'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Przychod',
    'ID_Klienta': 'Liczba_Klientow'
}).sort_values('Przychod', ascending=False)

print("Analiza miast:")
print(city_analysis)

# Wykres miast
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Liczba transakcji według miast
city_analysis['Liczba_Transakcji'].plot(kind='bar', ax=ax1, color='lightcoral')
ax1.set_title('Liczba Transakcji według Miast')
ax1.set_xlabel('Miasto')
ax1.set_ylabel('Liczba Transakcji')
ax1.tick_params(axis='x', rotation=45)

# Przychód według miast
city_analysis['Przychod'].plot(kind='bar', ax=ax2, color='lightsteelblue')
ax2.set_title('Przychód według Miast')
ax2.set_xlabel('Miasto')
ax2.set_ylabel('Przychód (PLN)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 9. ANALIZA KLIENTÓW
# ============================================================================

print("\n=== ANALIZA KLIENTÓW ===")

# Top klienci według wartości transakcji
top_customers = df_clean.groupby('ID_Klienta').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Wartosc_Zakupow'
}).sort_values('Wartosc_Zakupow', ascending=False).head(10)

print("Top 10 klientów według wartości zakupów:")
print(top_customers)

# Analiza częstotliwości zakupów
customer_frequency = df_clean.groupby('ID_Klienta')['ID_Transakcji'].count()
print(f"\nŚrednia liczba transakcji na klienta: {customer_frequency.mean():.2f}")
print(f"Maksymalna liczba transakcji na klienta: {customer_frequency.max()}")
print(f"Minimalna liczba transakcji na klienta: {customer_frequency.min()}")

# Wykres top klientów
plt.figure(figsize=(12, 6))
top_customers.head(10)['Wartosc_Zakupow'].plot(kind='bar', color='mediumpurple')
plt.title('Top 10 Klientów według Wartości Zakupów')
plt.xlabel('ID Klienta')
plt.ylabel('Wartość Zakupów (PLN)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================================
# 10. ANALIZA WARTOŚCI TRANSAKCJI
# ============================================================================

print("\n=== ANALIZA WARTOŚCI TRANSAKCJI ===")

# Statystyki wartości transakcji
print("Statystyki wartości transakcji:")
print(df_clean['Wartosc_Transakcji'].describe())

# Rozkład wartości transakcji
plt.figure(figsize=(12, 6))
plt.hist(df_clean['Wartosc_Transakcji'], bins=30, color='lightseagreen', alpha=0.7, edgecolor='black')
plt.title('Rozkład Wartości Transakcji')
plt.xlabel('Wartość Transakcji (PLN)')
plt.ylabel('Liczba Transakcji')
plt.grid(True, alpha=0.3)
plt.show()

# Box plot wartości transakcji według kategorii
plt.figure(figsize=(12, 6))
df_clean.boxplot(column='Wartosc_Transakcji', by='Kategoria', ax=plt.gca())
plt.title('Rozkład Wartości Transakcji według Kategorii')
plt.suptitle('')  # Usunięcie domyślnego tytułu
plt.xlabel('Kategoria')
plt.ylabel('Wartość Transakcji (PLN)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================================
# 11. ANALIZA SEZONOWOŚCI
# ============================================================================

print("\n=== ANALIZA SEZONOWOŚCI ===")

# Transakcje według dni tygodnia
df_clean['DzienTygodnia'] = df_clean['Data_Transakcji'].dt.day_name()
daily_analysis = df_clean.groupby('DzienTygodnia').agg({
    'ID_Transakcji': 'count',
    'Wartosc_Transakcji': 'sum'
}).rename(columns={
    'ID_Transakcji': 'Liczba_Transakcji',
    'Wartosc_Transakcji': 'Przychod'
})

# Sortowanie według dni tygodnia
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_analysis = daily_analysis.reindex(day_order)

print("Transakcje według dni tygodnia:")
print(daily_analysis)

# Wykres dni tygodnia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

daily_analysis['Liczba_Transakcji'].plot(kind='bar', ax=ax1, color='lightpink')
ax1.set_title('Liczba Transakcji według Dni Tygodnia')
ax1.set_xlabel('Dzień Tygodnia')
ax1.set_ylabel('Liczba Transakcji')
ax1.tick_params(axis='x', rotation=45)

daily_analysis['Przychod'].plot(kind='bar', ax=ax2, color='lightyellow')
ax2.set_title('Przychód według Dni Tygodnia')
ax2.set_xlabel('Dzień Tygodnia')
ax2.set_ylabel('Przychód (PLN)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 12. PODSUMOWANIE I WNIOSKI
# ============================================================================

print("\n" + "="*60)
print("PODSUMOWANIE I WNIOSKI")
print("="*60)

print(f"\n📊 OGÓLNE STATYSTYKI:")
print(f"   • Łączna liczba transakcji: {total_transactions}")
print(f"   • Łączny przychód: {total_revenue:,.2f} PLN")
print(f"   • Średnia wartość transakcji: {avg_transaction_value:,.2f} PLN")
print(f"   • Liczba unikalnych klientów: {unique_customers}")
print(f"   • Liczba unikalnych produktów: {unique_products}")

print(f"\n🏆 NAJPOPULARNIEJSZE KATEGORIE:")
top_category = category_analysis.iloc[0]
print(f"   • Najlepiej sprzedająca się kategoria: {top_category.name}")
print(f"   • Liczba transakcji: {top_category['Liczba_Transakcji']}")
print(f"   • Przychód: {top_category['Przychod']:,.2f} PLN")

print(f"\n🏙️ NAJLEPSZE MIASTA:")
top_city = city_analysis.iloc[0]
print(f"   • Miasto z największym przychodem: {top_city.name}")
print(f"   • Liczba transakcji: {top_city['Liczba_Transakcji']}")
print(f"   • Przychód: {top_city['Przychod']:,.2f} PLN")

print(f"\n📈 TRENDY CZASOWE:")
monthly_growth = monthly_transactions['Przychod'].pct_change() * 100
if len(monthly_growth) > 1:
    print(f"   • Wzrost przychodu między miesiącami: {monthly_growth.iloc[-1]:.1f}%")

print(f"\n💡 KLUCZOWE WNIOSKI:")
print(f"   1. Sklep odnotował {total_transactions} transakcji w Q1 2025")
print(f"   2. Łączny przychód wyniósł {total_revenue:,.0f} PLN")
print(f"   3. Najpopularniejszą kategorią są {top_category.name}")
print(f"   4. {top_city.name} generuje największe przychody")
print(f"   5. Średnia wartość transakcji to {avg_transaction_value:,.0f} PLN")

print("\n" + "="*60)
print("ANALIZA ZAKOŃCZONA")
print("="*60)

# ============================================================================
# 13. WYKRES W STYLU ML
# ============================================================================

def create_ml_style_plot(data1, data2, title="Time Series Plot"):
    """
    Tworzy wykres w stylu wykresu uczenia maszynowego
    """
    plt.figure(figsize=(12, 6))
    
    # Ustawienie stylu
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Ustawienie koloru tła
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('#ffffff')
    
    # Rysowanie linii
    plt.plot(data1, color='#4361ee', alpha=0.8, linewidth=1.5, label='Seria 1')
    plt.plot(data2, color='#4cc9f0', alpha=0.6, linewidth=2, label='Seria 2')
    
    # Konfiguracja siatki
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Konfiguracja etykiet
    plt.title(title, pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Oś X', labelpad=10)
    plt.ylabel('Oś Y', labelpad=10)
    
    # Legenda
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Marginesy
    plt.tight_layout()
    
    return plt

# Przykład użycia:
# Generowanie przykładowych danych
x = np.linspace(0, 100, 50)
y1 = np.sin(x/10) + np.random.normal(0, 0.1, 50)
y2 = np.sin(x/10) + 0.2

# Tworzenie wykresu
create_ml_style_plot(y1, y2, "Przykładowy Wykres ML")
plt.show()

# ============================================================================
# 14. PREDYKCJA CEN - MODEL ML
# ============================================================================

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_sequences(data, seq_length):
    """Przygotowuje sekwencje dla modelu LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_lstm_model(seq_length, dropout_rate=0.1):
    """Tworzy model LSTM do predykcji cen"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1), 
                           return_sequences=True),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LSTM(25, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Przygotowanie danych
print("\n=== PREDYKCJA CEN ===")

# Pobieranie danych o cenach
price_data = df_clean.groupby('Data')['Cena_Jednostkowa'].mean().reset_index()
prices = price_data['Cena_Jednostkowa'].values.reshape(-1, 1)

# Normalizacja danych
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Parametry modelu
seq_length = 5  # Długość sekwencji
dropout_rate = 0.1  # Współczynnik dropout

# Przygotowanie sekwencji
X, y = prepare_sequences(prices_scaled, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie i trenowanie modelu
model = create_lstm_model(seq_length, dropout_rate)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Predykcje
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Przygotowanie prognoz na przyszłość
last_sequence = prices_scaled[-seq_length:]
future_predictions = []
for _ in range(10):  # 10 przyszłych punktów
    next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
    future_predictions.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# Przekształcenie wartości z powrotem do oryginalnej skali
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Tworzenie indeksów dla wykresu
train_idx = price_data['Data'].iloc[seq_length:len(train_pred)+seq_length]
test_idx = price_data['Data'].iloc[len(train_pred)+seq_length:len(price_data)]
future_idx = pd.date_range(start=price_data['Data'].iloc[-1], periods=11, freq='D')[1:]

# Tworzenie wykresu
plt.figure(figsize=(15, 8))
plt.style.use('seaborn-v0_8-darkgrid')

# Ustawienie tła
plt.gca().set_facecolor('#f8f9fa')
plt.gcf().set_facecolor('#ffffff')

# Rysowanie linii
plt.plot(price_data['Data'], prices, color='#4361ee', alpha=0.8, linewidth=1.5, 
         label='Rzeczywiste ceny')
plt.plot(train_idx, train_pred, color='#4cc9f0', alpha=0.6, linewidth=2, 
         label='Predykcje (trening)')
plt.plot(test_idx, test_pred, color='#4cc9f0', alpha=0.6, linewidth=2, 
         label='Predykcje (test)')
plt.plot(future_idx, future_pred, color='#00b4d8', linestyle='--', alpha=0.8, 
         linewidth=2, label='Prognoza')

# Konfiguracja wykresu
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Predykcja Cen - Model LSTM', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Data', labelpad=10)
plt.ylabel('Cena', labelpad=10)
plt.legend(frameon=True, facecolor='white', framealpha=1)
plt.xticks(rotation=45)
plt.tight_layout()

# Wyświetlenie parametrów modelu
print("\nParametry modelu:")
print(f"Liczba jednostek LSTM: pierwsza warstwa - 50, druga warstwa - 25")
print(f"Współczynnik dropout: {dropout_rate}")
print(f"Długość sekwencji: {seq_length}")
print(f"\nWyniki:")
print(f"MSE na zbiorze treningowym: {history.history['loss'][-1]:.4f}")
print(f"MSE na zbiorze walidacyjnym: {history.history['val_loss'][-1]:.4f}")

plt.show()
