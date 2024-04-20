#empezamos con la edición del código
# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse

# Cargar los datos de la serie temporal (por ejemplo, datos de ventas mensuales)
data = pd.read_csv("C:/Users/esthe/Downloads/CEU/1ºCARRERA/Segundo semestre/Proyecto 1/Código ARIMA/datos_pasajeros.csv")
data = data.dropna()
pd.to_numeric(data)

data
# Visualizar los primeros registros de los datos
print(data.head())

# Graficar la serie temporal
data.plot()
plt.xlabel('Número de psajeros')
plt.ylabel('Valor')
plt.title('Serie Temporal')
plt.show()
