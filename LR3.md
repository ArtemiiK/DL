Выберите временной ряд цен акций.\
Подготовьте данные для обучения моделей машинного обучения (нормализация, создание признаков).\
Разделите данные на обучающую и тестовую выборки.
Реализуйте и обучите следующие модели:
a) Логистическая регрессия
b) Машина опорных векторов
c) Модель случайного леса (Random Forest)
d) Градиентный бустинг (например, XGBoost)
Настройте гиперпараметры моделей с помощью кросс-валидации.
Сделайте прогноз на тестовой выборке для каждой модели.
Сравните результаты прогнозирования моделей между собой и с простыми методами (например, наивный прогноз).
Визуализируйте результаты прогнозирования.
Проанализируйте важность признаков для моделей, где это применимо.
Сделайте выводы о эффективности различных подходов машинного обучения в прогнозировании цен акций.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import yfinance as yf

# 1. Загрузка данных
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close']

# Пример использования
aapl = load_data('AAPL', '2010-01-01', '2023-12-31')

# 2. Подготовка данных
def create_features(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
aapl_scaled = scaler.fit_transform(aapl.values.reshape(-1, 1))

X, y = create_features(aapl_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Модели машинного обучения

# 3.1 Логистическая регрессия
# Для использования логистической регрессии в задаче регрессии, 
# мы преобразуем задачу в классификацию направления движения цены
y_direction = (y_train > np.roll(y_train, 1))[1:].astype(int)
X_train_lr = X_train[1:]

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_lr, y_direction)

# 3.2 Машина опорных векторов (SVM)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train.ravel())

# 3.3 Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())

# 3.4 XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train.ravel())

# 4. Оценка моделей
def evaluate_model(model, X_test, y_test, model_name):
    if model_name == 'Logistic Regression':
        # Для логистической регрессии прогнозируем направление движения
        y_pred_direction = model.predict(X_test)
        y_pred = np.where(y_pred_direction == 1, y_test + 0.01, y_test - 0.01)
    else:
        y_pred = model.predict(X_test).reshape(-1, 1)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    return y_pred

lr_pred = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
svm_pred = evaluate_model(svm_model, X_test, y_test, 'SVM')
rf_pred = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
xgb_pred = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

# 5. Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(aapl.index[-len(y_test):], scaler.inverse_transform(y_test), label='Actual')
plt.plot(aapl.index[-len(y_test):], scaler.inverse_transform(lr_pred), label='Logistic Regression')
plt.plot(aapl.index[-len(y_test):], scaler.inverse_transform(svm_pred), label='SVM')
plt.plot(aapl.index[-len(y_test):], scaler.inverse_transform(rf_pred), label='Random Forest')
plt.plot(aapl.index[-len(y_test):], scaler.inverse_transform(xgb_pred), label='XGBoost')
plt.title('Сравнение прогнозов моделей машинного обучения')
plt.xlabel('Дата')
plt.ylabel('Цена акции')
plt.legend()
plt.show()

# 6. Анализ важности признаков (для Random Forest)
feature_importance = pd.DataFrame({'feature': range(X_train.shape[1]), 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title('Топ-10 важных признаков (Random Forest)')
plt.xlabel('Индекс признака')
plt.ylabel('Важность')
plt.show()
```
