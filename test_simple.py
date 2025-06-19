import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Rozpoczynam test...")

for i in range(3):
    print(f"Test {i+1}")
    
    # Tworzenie prostego wykresu
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + i * 0.5
    ax.plot(x, y)
    ax.set_title(f'Test wykresu {i+1}')
    
    # Zapisywanie
    filename = f'test_{i+1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Zapisano {filename}")
    plt.close()

print("Test zakończony!") 