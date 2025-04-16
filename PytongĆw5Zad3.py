import pylab
import numpy as np

def main():
    tytul = input("Podaj tytuł wykresu: ")
    a = float(input("Podaj współczynnik kierunkowy funkcji (a): "))
    b = float(input("Podaj wyraz wolny funkcji (b): "))
    x_min = float(input("Podaj minimalną wartość x: "))
    x_max = float(input("Podaj maksymalną wartość x: "))
    czy_siatka = input("Czy wyświetlić siatkę pomocniczą? (tak/nie): ").lower() == "tak"
   
    x = np.linspace(x_min, x_max, 100)
    y = a * x + b
   
    pylab.plot(x, y)
    pylab.title(tytul)
    pylab.xlabel('Oś X')
    pylab.ylabel('Oś Y')
    pylab.grid(czy_siatka)
    pylab.annotate(f'f(x) = {a}x + {b}', xy=(0.05, 0.95), xycoords='axes fraction')
    pylab.show()

if __name__ == "__main__":
    main()