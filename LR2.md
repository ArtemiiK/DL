Выберите временной ряд с ценами актива (предпочтительно с явной сезонностью, например, акции компании с сезонным бизнесом).\
Проведите декомпозицию временного ряда на тренд, сезонность и остаток.
Визуализируйте компоненты декомпозиции.
Проанализируйте сезонность: определите периодичность и амплитуду сезонных колебаний.
Удалите сезонность из ряда и сравните исходный ряд с десезонализированным.
Постройте прогноз на основе выявленных тренда и сезонности на 12 периодов вперед.
Оцените качество прогноза, используя метрики MAE и RMSE.
Сделайте выводы о влиянии сезонности на цены актива и эффективности прогнозирования.
Проведите тест на стационарность ряда (тест Дики-Фуллера).
Если ряд нестационарен, приведите его к стационарному виду.
Постройте и сравните модели AR, MA, ARMA и ARIMA.
Подберите оптимальные параметры для каждой модели, используя информационные критерии (AIC, BIC).
Проведите диагностику остатков моделей.
Сделайте прогноз на 30 дней вперед для каждой модели.
Сравните качество прогнозов моделей, используя метрики MAE, RMSE и MAPE.
Визуализируйте результаты прогнозирования.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

# 1. Загрузка данных
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

# Пример использования
btc_usd = load_data('BTC-USD', '2020-01-01', '2023-12-31')

# 2. Визуализация временного ряда
plt.figure(figsize=(12, 6))
plt.plot(btc_usd)
plt.title('Цена закрытия BTC-USD')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.show()

# 3. Декомпозиция временного ряда
decomposition = seasonal_decompose(btc_usd, model='additive', period=30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Наблюдаемый')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Тренд')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Сезонность')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Остаток')
plt.tight_layout()
plt.show()

# 4. Тест на стационарность
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

adf_test(btc_usd)

# 5. Построение и оценка моделей ARIMA
def evaluate_arima_model(X, arima_order):
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * 0.8)
    train, test = X[:train_size], X[train_size:]
    
    # Обучение модели
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    
    # Прогноз
    forecast = model_fit.forecast(steps=len(test))
    
    # Оценка качества прогноза
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    return mae, rmse

# Перебор параметров ARIMA
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

best_score, best_cfg = float("inf"), None
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                mae, rmse = evaluate_arima_model(btc_usd, order)
                if mae < best_score:
                    best_score, best_cfg = mae, order
                print(f'ARIMA{order} MAE={mae:.3f} RMSE={rmse:.3f}')
            except:
                continue

print(f'Лучшая модель ARIMA{best_cfg} MAE={best_score:.3f}')

# 6. Прогнозирование с использованием лучшей модели
model = ARIMA(btc_usd, order=best_cfg)
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

plt.figure(figsize=(12, 6))
plt.plot(btc_usd.index, btc_usd, label='Исторические данные')
plt.plot(forecast.index, forecast, color='red', label='Прогноз')
plt.title('Прогноз цены BTC-USD на 30 дней')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()

# 7. Вывод результатов
print(model_fit.summary())
print("\nПрогноз на 30 дней:")
print(forecast)
```
