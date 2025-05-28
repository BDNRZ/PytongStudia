import tkinter as tk
from tkinter import *
from tkinter import ttk
import requests
import json
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

def find_similar_pattern(df, pattern_length=10, max_deviation=0.05):
    recent_pattern = df.iloc[-pattern_length:].copy()
    
    recent_changes = np.array([
        recent_pattern['close'].pct_change().fillna(0).values,
        recent_pattern['open'].pct_change().fillna(0).values
    ]).T
    
    for i in range(len(df) - 2*pattern_length):
        historical_segment = df.iloc[i:i+pattern_length].copy()
        
        historical_changes = np.array([
            historical_segment['close'].pct_change().fillna(0).values,
            historical_segment['open'].pct_change().fillna(0).values
        ]).T
        
        mae = np.mean(np.abs(recent_changes - historical_changes))
        
        if mae < max_deviation:
            return i, i+pattern_length
    
    return None

def fetch_and_display_candles(symbol, interval, limit, pattern_mode):
    url = f'https://www.mexc.com/open/api/v2/market/kline?symbol={symbol}&interval={interval}&limit={limit}'
    
    try:
        response = requests.get(url)
        data = json.loads(response.text)["data"]
        formatted_data = format_candles_data(data)
        
        df = pd.DataFrame(formatted_data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index("time")
        
        if pattern_mode == 'PATTERN_SEARCH':
            result = find_similar_pattern(df)
            
            if result:
                start_idx, end_idx = result
                df_to_plot = df.iloc[start_idx:end_idx]
                title = 'Znaleziono podobny wzorzec'
            else:
                df_to_plot = df.iloc[-10:]
                title = 'Ostatni wzorzec (Nie znaleziono podobnych)'
        else:
            df_to_plot = df
            title = f'Wykres świecowy dla {symbol}'
        
        # Create custom style
        mc = mpf.make_marketcolors(up='green', down='red',
                                 edge='inherit',
                                 wick='inherit',
                                 volume='in',
                                 ohlc='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, 
                              gridstyle='',
                              rc={'figure.figsize': (20, 12)})  # Set figure size in style
        
        kwargs = {
            'type': 'candle',
            'title': title,
            'style': s,
            'volume': False,  # Disable volume to give more space to candles
            'tight_layout': False,  # Use tight layout to maximize space
            'returnfig': True
        }
        
        fig, _ = mpf.plot(
            df_to_plot,
            **kwargs
        )
        
        return fig
    except Exception as e:
        print(f"Błąd: {e}")
        return None

class CandlestickApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Analiza Wykresów Świecowych')
        
        # Set window size
        self.root.geometry('480x640')  # Set default window size to 480x640 (portrait)
        
        # Load background image
        self.background_image = PhotoImage(file='altum.png')
        self.background_label = Label(root, image=self.background_image)
        self.background_label.place(relwidth=1, relheight=1)
        
        # Create main frame
        self.frame = Frame(root, bg='#ffffff')
        self.frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.25, anchor='n')
        
        # Symbol input
        self.symbol_label = Label(self.frame, text="Symbol", font=('Arial', 10))
        self.symbol_label.place(relx=0.05, rely=0.1, relwidth=0.3, relheight=0.2)
        self.symbol_entry = Entry(self.frame, font=('Arial', 10))
        self.symbol_entry.place(relx=0.4, rely=0.1, relwidth=0.5, relheight=0.2)
        self.symbol_entry.insert(0, "GLQ_USDT")
        
        # Interval input
        self.interval_label = Label(self.frame, text="Interwał", font=('Arial', 10))
        self.interval_label.place(relx=0.05, rely=0.4, relwidth=0.3, relheight=0.2)
        self.interval_entry = Entry(self.frame, font=('Arial', 10))
        self.interval_entry.place(relx=0.4, rely=0.4, relwidth=0.5, relheight=0.2)
        self.interval_entry.insert(0, "1m")
        
        # Limit input
        self.limit_label = Label(self.frame, text="Limit świeczek", font=('Arial', 10))
        self.limit_label.place(relx=0.05, rely=0.7, relwidth=0.3, relheight=0.2)
        self.limit_entry = Entry(self.frame, font=('Arial', 10))
        self.limit_entry.place(relx=0.4, rely=0.7, relwidth=0.5, relheight=0.2)
        self.limit_entry.insert(0, "500")
        
        # Mode selection frame
        self.mode_frame = Frame(root, bg='#ffffff')
        self.mode_frame.place(relx=0.5, rely=0.4, relwidth=0.75, relheight=0.08, anchor='n')
        
        # Mode selection
        self.selected_mode = StringVar(root, 'NORMAL')
        self.normal_mode = Radiobutton(self.mode_frame, text="Tryb normalny", 
                                     variable=self.selected_mode, value='NORMAL',
                                     font=('Arial', 10))
        self.normal_mode.place(relx=0.1, rely=0.2, relwidth=0.4, relheight=0.6)
        
        self.pattern_mode = Radiobutton(self.mode_frame, text="Wyszukiwanie wzorców", 
                                      variable=self.selected_mode, value='PATTERN_SEARCH',
                                      font=('Arial', 10))
        self.pattern_mode.place(relx=0.5, rely=0.2, relwidth=0.4, relheight=0.6)
        
        # Display button
        self.display_button = Button(root, text="Wyświetl wykres", 
                                   command=self.display_chart,
                                   font=('Arial', 10))
        self.display_button.place(relx=0.5, rely=0.5, relwidth=0.3, relheight=0.04, anchor='n')
        
        # Chart frame
        self.chart_frame = Frame(root)
        self.chart_frame.place(relx=0.5, rely=0.6, relwidth=0.8, relheight=0.35, anchor='n')
        
    def display_chart(self):
        symbol = self.symbol_entry.get()
        interval = self.interval_entry.get()
        limit = self.limit_entry.get()
        mode = self.selected_mode.get()
        
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Fetch and display new chart
        fig = fetch_and_display_candles(symbol, interval, limit, mode)
        if fig:
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

if __name__ == "__main__":
    root = Tk()
    app = CandlestickApp(root)
    root.mainloop() 