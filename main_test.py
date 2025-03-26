import random
#import re
import string
import pandas as pd  # type: ignore
def serial_number_giver(lenght=12):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(lenght))

df = pd.read_csv('test_list_copy.csv')

initial_rows = len(df)

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
    if new_ship == "":
        break
    amount = input("Podaj ilość nowych statków:")
    new_row = {"Name": new_ship, "Serial Number": "", "Amount": amount}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    

df['Serial Number'] = df.apply(
    lambda row: row['Serial Number'] if pd.notna(row['Serial Number']) and str(row['Serial Number']).strip() != ''
                else (serial_number_giver() if pd.notna(row['Name']) and str(row['Name']).strip() != '' else None),
    axis=1
)
df['Amount'] = df['Amount'].apply(
    lambda x: x if pd.notna(x) and str(x).strip() != '' else amount)

df.to_csv('test_list_copy.csv', index=False)
added_rows = df.iloc[initial_rows:]
print("Dodano następujące statki:")
print(added_rows[['Name', 'Serial Number', 'Amount']].to_string(index=False))