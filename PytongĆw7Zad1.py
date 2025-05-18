import pandas as pd

def load_movies_data():
    return pd.read_csv('movies.csv', sep=';', encoding="ISO-8859-1", skiprows=[1])

def zadanie1():
    df = load_movies_data()
    latest_year = df['Year'].max()
    latest_movies = df[df['Year'] == latest_year]
    print(f"ZADANIE 1:")
    if len(latest_movies) > 0:
        print(latest_movies[['Year', 'Title', 'Director']])
        print(f"Total: {len(latest_movies)}")
    else:
        print(f"No movies from {latest_year} found.")

def zadanie2():
    df = load_movies_data()
    df = df.dropna(subset=['Length'])
    avg_lengths = df.groupby('Director')['Length'].mean().reset_index()
    avg_lengths = avg_lengths.sort_values(by='Director')
    print("\nZADANIE 2:")
    print(avg_lengths.head(10))
    print(f"Total: {len(avg_lengths)}")

def zadanie3():
    df = load_movies_data()
    selected_columns = df[['Title', 'Director', 'Popularity']]
    selected_columns.to_csv('selected_movies.csv', index=False)
    print("\nZADANIE 3:")
    print(selected_columns.head(5))

def zadanie4():   
    df = load_movies_data()
    awards_count = df[df['Awards'] == 'Yes'].shape[0]
    total_movies = df.shape[0]
    percentage = (awards_count / total_movies) * 100
    print("\nZADANIE 4:")
    print(f"Percentage: {percentage:.2f}%")
    print(f"Total: {total_movies}, With awards: {awards_count}")

def zadanie5():
    df = load_movies_data()
    kubrick_movies = df[df['Director'].str.contains('Kubrick', case=False, na=False)]
    print("\nZADANIE 5:")
    if len(kubrick_movies) > 0:
        for _, movie in kubrick_movies.iterrows():
            print(f"Year: {movie['Year']}, Title: {movie['Title']}, Awards: {movie['Awards']}")
        print(f"Total: {len(kubrick_movies)}")
    else:
        print("No Kubrick movies found.")

def zadanie6():
    df = load_movies_data()
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')
    comedy_movies = df[df['Subject'] == 'Comedy'].copy()
    popularity_sum = comedy_movies['Popularity'].sum()
    print("\nZADANIE 6:")
    print(f"Sum: {popularity_sum:.1f}")
    print(f"Count: {len(comedy_movies)}")

if __name__ == "__main__":
    zadanie1()
    zadanie2()
    zadanie3()
    zadanie4()
    zadanie5()
    zadanie6()