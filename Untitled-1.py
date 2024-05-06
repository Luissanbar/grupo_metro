import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse

data = pd.read_csv("/Users/marcofernandez/Desktop/grupo metro/datos_pasajeros.csv")
data = data.dropna()

data
print(data.head())

data.plot()
plt.xlabel('num pas')
plt.ylabel('valor')
plt.title('Serie Temporal')
plt.show()



train_size = int(len((data)) * 0.8)
train, test = data[:train_size], data[train_size:]
train
test
test.tail(10)
order = (5, 1, 0)  
model = ARIMA(data, order=order)
model_fit = model.fit()
model_fit.forecast(steps=3)

print(model_fit.summary())

test
predictions = model_fit.forecast(steps=len(test))
print(predictions)

rmse_value = rmse(test, predictions)
print('Error cuadrático medio (RMSE):', rmse_value)

plt.figure(figsize=(10, 6))

# Gráfico de los valores observados
plt.subplot(2, 1, 1)
plt.plot(test.index, test, label='Observado', color='blue')
plt.xlabel('Tiempo')
plt.ylabel('Número de pasajeros')
plt.title('Valores Observados')
plt.legend()

# Gráfico de las predicciones
plt.subplot(2, 1, 2)
plt.plot(test.index, predictions, label='Predicción', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Número de pasajeros')
plt.title('Predicciones ARIMA')
plt.legend()
plt.xlim(1747,1760)

plt.tight_layout()
plt.show()
