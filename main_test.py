import random
#import re
import string
import pandas as pd  # type: ignore
def serial_number_giver(lenght=12):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(lenght))

df = pd.read_csv('test_list_copy.csv')

if 'Serial Number' not in df.columns:
    df['Serial Number'] = None

if 'Name' not in df.columns:
    df['Name'] = None

if 'Amount' not in df.columns:
    df['Amount'] = None

def get_serial_number(x):
    if pd.notna(x) and str(x).strip():
        return x
    return serial_number_giver()

while True:
    new_ship = input("Podaj nazwę nowego statku/statków (Zostaw puste aby zapisać dodane statki)").strip()
    new_row = {"Name": new_ship}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    amount = input("Podaj ilość nowych droidów:")
    new_row = {"Amount": amount}

df['Serial Number'] = df['Serial Number'].apply(
    lambda x: x if pd.notna(x) and str(x).strip() != '' else serial_number_giver())

df.to_csv('test_list_copy.csv', index=False)

