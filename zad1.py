import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from matplotlib.lines import Line2D
import warnings
from scipy.stats import norm, cauchy
import matplotlib

# Ta opcja wyłącza interaktywne wykresy ponieważ powodowały (u nas) wyświetlanie wielu błędów w terminalu
matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


TOTAL_NUMBER_OF_RESPONDERS = 854

incom_mappings = {
    "< 5 tys.": 2.5,
    "5-10 tys.": 7.5,
    "10-20 tys.": 15,
    "20-50 tys.": 35,
    "50-100 tys.": 75,
    "> 100 tys.": 125,
}

investment_mappings = {
    "< 10 tys.": 5,
    "30-50 tys.": 40,
    "50-100 tys.": 75,
    "100-500 tys.": 300,
    "500-750 tys.": 625,
    "Powyżej 1 mln": 1250,
}

data = {
    "< 5 tys.": [46, 47, 17, 19, 3, 2],
    "5-10 tys.": [86, 99, 50, 62, 7, 2],
    "10-20 tys.": [38, 83, 67, 81, 26, 8],
    "20-50 tys.": [4, 19, 13, 26, 20, 10],
    "50-100 tys.": [0, 0, 0, 4, 4, 9],
    "> 100 tys.": [0, 0, 0, 0, 0, 2],
}
index = [
    "< 10 tys.",
    "30-50 tys.",
    "50-100 tys.",
    "100-500 tys.",
    "500-750 tys.",
    "Powyżej 1 mln",
]

investment_df = pd.DataFrame(data, index=index)
df_percent_of_people_with_income_investments = (
    investment_df.div(investment_df.sum(axis=1), axis=0) * 100
)
df_percent_of_people_with_income_investments = (
    df_percent_of_people_with_income_investments.round(1)
)

### Wykresy słupkowe pokazujące ile osób posiadających dany portfel investycyjny zarabia daną ilość pieniędy

investment_df.plot.bar(stacked=True, figsize=(12, 9))
plt.title(
    "Ilość osób w różnych przedziałach dochodowych, podzielona na wielkość portfela inwestycyjnego"
)
plt.xlabel("Wielkość portfela inwestycyjnego")
plt.ylabel("Liczba osób")
plt.xticks(rotation=30)
plt.savefig("task_1_portfel_inwestycyjny.png", bbox_inches="tight")
df_percent_of_people_with_income_investments.plot.bar(stacked=True, figsize=(12, 9))

plt.title(
    "Procentowy stosunek ilości osób z róznych przedziałów dochodowych, podzielona ze względu na wielkość portfela inwestycyjnego"
)
plt.xlabel("Wielkość portfela inwestycyjnego")
plt.ylabel("Procent osób")
plt.xticks(rotation=30)
plt.savefig("task_1_portfel_inwestycyjny_procent.png", bbox_inches="tight")


### Regresja liniowa z użyciem modułu statsmodels

# Rozszerzenie danych do wierszu na jedną osobę
df_long = (
    investment_df.reset_index()
    .rename(columns={"index": "investment"})
    .melt(id_vars="investment", var_name="earning", value_name="n")
)

df_long["earning_aprox"] = df_long["earning"].map(incom_mappings)
df_long["investment_aprox"] = df_long["investment"].map(investment_mappings)


X = sm.add_constant(df_long["earning_aprox"])
y = df_long["investment_aprox"]
weights = df_long["n"]

model = sm.WLS(
    y,
    X,
    weights=weights,
).fit()
# To powyżej jest równoznaczne z tym poniżej
# model = smf.wls("investment_aprox ~ earning_aprox", data=df_long, weights=df_long["n"]).fit()


# Podsumowanie Regresji
print(model.summary())


# Wykres reprezentujący modell regresji
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_long["earning_aprox"],
    df_long["investment_aprox"],
    s=df_long["n"] * 10,
    alpha=0.5,
    c=df_long["n"],
    cmap="viridis",
    label="Liczba gospodarstw",
)
plt.grid()

predicted = model.predict(X)
plt.plot(df_long["earning_aprox"], predicted, color="red", label="Linia regresji")

plt.xlabel("Średni miesięczny dochód (tys. zł)")
plt.ylabel("Średnia wartość portfela inwestycyjnego (tys. zł)")
plt.title("Regresja: Dochód a wartość inwestycji")
plt.colorbar(scatter, label="Liczba gospodarstw")
plt.legend()
plt.tight_layout()
plt.savefig("task_1_regresion.png", bbox_inches="tight")


### 3D Histogram, wraz z dopasowaniem dwóch rozkładów (normalny i Cauchy'ego)
dx, dy = 1.0, 0.4

df_percent_of_investments = investment_df.div(investment_df.sum(axis=0), axis=1) * 100
df_percent_of_investments = df_percent_of_investments.T


xpos = np.arange(df_percent_of_investments.shape[1])
ypos = np.arange(df_percent_of_investments.shape[0])

xpos_grid, ypos_grid = np.meshgrid(xpos, ypos)
xpos = xpos_grid.flatten()
ypos = ypos_grid.flatten()

zpos = np.zeros_like(xpos)
dz = df_percent_of_investments.values.flatten()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
x_ticks = np.arange(df_percent_of_investments.shape[1])

means = []
stds = []
medians = []

# Dopasowanie rozkładów normalnego i Cauchy'ego
for i, row in enumerate(df_percent_of_investments.values):
    x_vals = x_ticks
    y_vals = row
    mu, std = norm.fit(np.repeat(x_vals, (y_vals * 10).astype(int)))
    cauchy_mu, cauchy_std = cauchy.fit(np.repeat(x_vals, (y_vals * 10).astype(int)))

    if std < 0.1:
        std = 0.1
    x_fit = np.linspace(x_vals.min(), x_vals.max() + 1, 200)
    y_fit = norm.pdf(x_fit, mu, std)
    y_fit = y_fit / y_fit.max() * row.max()
    ax.plot(x_fit, np.full_like(x_fit, i), y_fit, color="red", lw=2, zorder=10)
    y_fit_cauchy = cauchy.pdf(x_fit, cauchy_mu, cauchy_std)
    y_fit_cauchy = y_fit_cauchy / y_fit_cauchy.max() * row.max()
    ax.plot(
        x_fit,
        np.full_like(x_fit, i),
        y_fit_cauchy,
        color="blue",
        lw=2,
        zorder=10,
    )

custom_lines = [
    Line2D([0], [0], color="red", lw=4),
    Line2D([0], [0], color="blue", lw=4),
]
ax.legend(custom_lines, ["Normalny", "Cauchego"], loc="upper right")

ax.set_xticks(np.arange(len(index)))
ax.set_xticklabels(index)
ax.set_yticks(np.arange(len(data) + 1))
ax.set_yticklabels([""] + list(data.keys()), rotation=10)


ax.set_zlabel("% Osób z danego przedziału dochodowego")
ax.set_xlabel("Portfel inwestycyjny (tys. zł)")
ax.xaxis.labelpad = 10
ax.set_ylabel("Dochód (tys. zł)")
ax.yaxis.labelpad = 10
plt.tight_layout()
plt.title(
    "Rozkład procentowy osób w różnych przedziałach dochodowych, podzielona na wielkość portwela inwestycyjnego"
)
plt.savefig("task_1_portfel_inwestycyjny_3d.png", bbox_inches="tight")


### Doadatkowe statystyki

df_expanded = df_long.loc[df_long.index.repeat(df_long["n"])].reset_index(drop=True)
df_expanded = df_expanded[["investment", "earning"]]
df_expanded["investment"] = df_expanded["investment"].map(investment_mappings)
df_expanded["earning"] = df_expanded["earning"].map(incom_mappings)
print()
print(f"Średnia inwestycji: {df_expanded['investment'].mean():.2f} tys. zł")
print(
    f"Odchylenie standardowe inwestycji: {df_expanded['investment'].std():.2f} tys. zł"
)
print(f"Mediana inwestycji: {df_expanded['investment'].median():.2f} tys. zł")
print(f"Średni dochód: {df_expanded['earning'].mean():.2f} tys. zł")
print(f"Odchylenie standardowe dochodu: {df_expanded['earning'].std():.2f} tys. zł")
print(f"Mediana dochodu: {df_expanded['earning'].median():.2f} tys. zł")
print()
print("Średnia, odchylenie standardowe i mediana dla każdego przedziału dochodowego:")
print("Means:", list(df_expanded.groupby("earning")["investment"].mean()))
print("Stds:", list(df_expanded.groupby("earning")["investment"].std()))
print("Medians:", list(df_expanded.groupby("earning")["investment"].median()))

plt.figure(figsize=(12, 6))
df_expanded.groupby("earning")["investment"].mean().plot()
plt.ylabel("Średnia wartość inwestycji (tys. zł)")
plt.xlabel("Mesięczny dochód (tys. zł)")
plt.title("Średnia wartość inwestycji w zależności od dochodu")
plt.savefig("task_1_srednia_wartosc_inwestycji.png", bbox_inches="tight")

plt.show()
