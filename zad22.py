import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

today = datetime.now().strftime("%Y-%m-%d")

sp500 = yf.Ticker("^GSPC")
data = sp500.history(period="5y")
closes = data['Close']

rsi = calculate_rsi(closes, periods=14)

position = False
entry_price = 0
entry_date = None
returns = []
equity = [10000]
dates = []
trade_log = []

for i in range(14, len(closes) - 126):
    if not position:
        if rsi.iloc[i] < 40:
            position = True
            entry_price = closes.iloc[i]
            entry_date = closes.index[i]
    elif position:
        current_price = closes.iloc[i]
        profit = (current_price - entry_price) / entry_price
        days_held = (closes.index[i] - entry_date).days
        
        if profit >= 0.10 or profit <= -0.05 or days_held >= 126 or rsi.iloc[i] > 70:
            position = False
            returns.append(profit)
            equity.append(equity[-1] * (1 + profit))
            trade_log.append({
                'Entry Date': entry_date,
                'Exit Date': closes.index[i],
                'Entry Price': entry_price,
                'Exit Price': current_price,
                'Profit': profit,
                'Days Held': days_held
            })
            dates.append(closes.index[i])

total_return = (equity[-1] / equity[0] - 1) if equity else 0
annualized_return = ((1 + total_return) ** (252 / len(closes)) - 1) if total_return else 0
win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
avg_profit = np.mean(returns) if returns else 0

print("Wyniki strategii opartej na RSI:")
print(f"Całkowity zwrot: {total_return:.2%}")
print(f"Zwrot annualized: {annualized_return:.2%}")
print(f"Współczynnik wygranych: {win_rate:.2%}")
print(f"Średni zysk na transakcję: {avg_profit:.2%}")
print(f"Liczba transakcji: {len(trade_log)}")
print("\nLog transakcji:")
for trade in trade_log:
    print(f"Wejście: {trade['Entry Date'].date()} | Wyjście: {trade['Exit Date'].date()} | "
          f"Zysk: {trade['Profit']:.2%} | Dni: {trade['Days Held']}")

plt.figure(figsize=(12, 6))
plt.plot(dates, equity[1:], label='Kapitał strategii', color='b')
plt.title('Krzywa kapitału strategii opartej na RSI')
plt.xlabel('Data')
plt.ylabel('Kapitał (USD)')
plt.legend()
plt.grid()
plt.savefig(f'equity_curve_{today}.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(closes.index, closes, label='S&P 500', color='k', alpha=0.5)
for trade in trade_log:
    plt.axvline(x=trade['Entry Date'], color='g', linestyle='--', alpha=0.5, label='Kupno' if trade == trade_log[0] else "")
    plt.axvline(x=trade['Exit Date'], color='r', linestyle='--', alpha=0.5, label='Sprzedaż' if trade == trade_log[0] else "")
plt.title('Ceny S&P 500 z sygnałami kupna/sprzedaży')
plt.xlabel('Data')
plt.ylabel('Cena zamknięcia')
plt.legend()
plt.grid()
plt.savefig(f'sp500_signals_{today}.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(returns, bins=30, density=True, alpha=0.6, color='b', label='Zyski z transakcji')
plt.title('Rozkład zysków z transakcji')
plt.xlabel('Zysk')
plt.ylabel('Gęstość')
plt.legend()
plt.grid()
plt.savefig(f'returns_histogram_{today}.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(closes.index, rsi, label='RSI', color='purple')
plt.axhline(y=40, color='g', linestyle='--', alpha=0.5, label='RSI 40 (wejście)')
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='RSI 70 (wyjście)')
plt.title('RSI dla S&P 500')
plt.xlabel('Data')
plt.ylabel('RSI')
plt.legend()
plt.grid()
plt.savefig(f'rsi_plot_{today}.png')
plt.close()