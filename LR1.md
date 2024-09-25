
* Найдите временной ряд с ценами активов (цены акций, облигаций, нефти, криптовалюты и т.д.)
* Отобразите временной ряд на графике
* Оцените его
* Сделайте начальные выводы
* Постройте 4 скользящие средние с окнами (5, 10, 30, 50)
* Оцените график с скоьзящими среднмии
* Сделайте предыдыдущие 2 пункта с экспоненциальными скользящими средними
* Сделайте прогноз движения актива на основе экспоненциального скользящего среднего на 5 периодов вперед.
* Сделайте выводы по проделанной работе и опипшите их.

'''python import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных временного ряда
data = pd.read_csv("data.csv", parse_dates=True, index_col="Date")

# График временного ряда
plt.plot(data.index, data["Value"])
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.title("График временного ряда")
plt.show()

# График скользящего среднего
rolling_mean = data["Value"].rolling(window=7).mean()
plt.plot(data.index, data["Value"], label="Исходные значения")
plt.plot(data.index, rolling_mean, label="Скользящее среднее")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.title("График скользящего среднего")
plt.legend()
plt.show()

# График автокорреляции
pd.plotting.autocorrelation_plot(data["Value"])
plt.xlabel("Лаг")
plt.ylabel("Корреляция")
plt.title("График автокорреляции")
plt.show()

# График разложения временного ряда
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data["Value"], model="additive")
decomposition.plot()
plt.xlabel("Дата")
plt.suptitle("Разложение временного ряда")
plt.show()
'''

