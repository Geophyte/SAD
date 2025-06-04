import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, kstest

# Pobieranie danych S&P 500
sp500 = yf.Ticker("^GSPC")
data = sp500.history(period="5y")
closes = data['Close']

# Obliczenie 6-miesięcznych zysków
window = 126  # 6 miesięcy ≈ 126 dni handlowych
z_t0 = (closes.shift(-window) - closes) / closes
z_t0 = z_t0.dropna()

# a) Histogram 6-miesięcznych zysków
plt.figure(figsize=(10, 6))
plt.hist(z_t0, bins=50, density=True, alpha=0.6, color='b', label='6-miesięczne zyski')

# b) Dopasowanie rozkładu skośno-normalnego
params = skewnorm.fit(z_t0)  # Wyestymowanie parametrów: a (skośność), loc (lokalizacja), scale (skala)
a, loc, scale = params
x = np.linspace(min(z_t0), max(z_t0), 100)
plt.plot(x, skewnorm.pdf(x, a, loc, scale), 'k-', lw=2, label='Rozkład skośno-normalny')
plt.title('Histogram 6-miesięcznych zysków S&P 500 z dopasowaniem skośno-normalnym')
plt.xlabel('6-miesięczne zyski')
plt.ylabel('Gęstość')
plt.legend()
plt.savefig("dopasowanie_rozkladu_skosno_normalnego.png")

# c) Wyestymowanie parametrów rozkładu
print("Parametry rozkładu skośno-normalnego dla 6-miesięcznych zysków:")
print(f"Skośność (a): {a:.6f}")
print(f"Lokalizacja (loc): {loc:.6f}")
print(f"Skala (scale): {scale:.6f}")

# d) Sprawdzenie dopasowania rozkładu
ks_stat_z, ks_p_z = kstest(z_t0, 'skewnorm', args=params)
print("\nTest dobroci dopasowania (Kolmogorov-Smirnov):")
print(f"Statystyka KS: {ks_stat_z:.6f}, p-wartość: {ks_p_z:.6f}")