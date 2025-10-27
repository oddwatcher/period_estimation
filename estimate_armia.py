import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import savgol_filter
from pathlib import Path

date_strings = Path("date_strings.txt").read_text().splitlines()

"""
examples of data_strings.txt:
2023.2.25 
2023.3.29
2023.4.29
2023.6.2
2023.7.4
2023.8.9
2023.9.11
2023.10.9
2023.11.9
2023.12.13
2024.1.20
2024.2.19
"""
dates = sorted([datetime.strptime(d, "%Y.%m.%d") for d in date_strings])

intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
x = np.arange(1, len(intervals) + 1)
y = np.array(intervals)

poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_model.fit(x.reshape(-1, 1), y)
x_poly = np.linspace(1, len(x), 100)
y_poly = poly_model.predict(x_poly.reshape(-1, 1))

y_smooth = savgol_filter(y, window_length=7, polyorder=2)

ts = pd.Series(y, index=pd.date_range(start="2022-10-01", periods=len(y), freq="D"))

model = ARIMA(ts, order=(5, 1, 2))
results = model.fit()

future_steps = 5
arima_forecast = results.get_forecast(steps=future_steps)
y_future = arima_forecast.predicted_mean.values
conf_int = arima_forecast.conf_int()

last_date = dates[-1]
future_dates = []
current = last_date
for i in range(future_steps):
    current += timedelta(days=int(round(y_future[i])))
    future_dates.append(current.strftime("%Y.%m.%d"))

plt.figure(figsize=(14, 8))

plt.plot(
    x,
    y,
    "o-",
    color="#1f77b4",
    linewidth=1.5,
    markersize=5,
    alpha=0.8,
    label="Actual Intervals",
)

plt.plot(x, y_smooth, "k-", linewidth=2, alpha=0.7, label="Trend (Savitzky-Golay)")

plt.plot(
    x_poly,
    y_poly,
    "r--",
    linewidth=2,
    alpha=0.7,
    label="Long-term Trend (Cubic Polynomial)",
)

x_future = np.arange(len(x) + 1, len(x) + 1 + future_steps)
plt.plot(
    x_future,
    y_future,
    "s--",
    color="#d62728",
    linewidth=2.5,
    markersize=8,
    label="ARIMA Prediction",
)

plt.fill_between(
    x_future,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="#d62728",
    alpha=0.2,
    label="95% Confidence Interval",
)

for i, val in enumerate(y):
    if i % 3 == 0:
        plt.text(
            x[i],
            val + 1,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

for i, val in enumerate(y_future):
    plt.text(
        x_future[i],
        val + 1,
        f"{int(round(val))}",
        ha="center",
        va="bottom",
        color="#d62728",
        fontsize=10,
        fontweight="bold",
    )

plt.title("Date Interval Analysis with Advanced Forecasting", fontsize=16, pad=20)
plt.xlabel("Interval Number", fontsize=14)
plt.ylabel("Days Between Dates", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper left", fontsize=11)

r2 = r2_score(y, poly_model.predict(x.reshape(-1, 1)))
plt.figtext(
    0.5,
    0.01,
    f"Model Performance | R²: {r2:.3f} | Future Dates: {', '.join(future_dates)}\n"
    f"Note: Red dashed line = ARIMA prediction (5 intervals), Shaded area = 95% confidence interval",
    ha="center",
    fontsize=10,
    bbox=dict(
        facecolor="lightyellow", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.5"
    ),
)

trend_desc = "Long-term trend shows slight increase in interval lengths\n"
trend_desc += "Short-term fluctuations smoothed using Savitzky-Golay filter"
plt.figtext(
    0.15,
    0.25,
    trend_desc,
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
