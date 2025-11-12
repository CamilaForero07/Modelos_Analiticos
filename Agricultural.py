#%%
# ANALISIS EXPLORATORIO EDA - AGRICULTURAL
from opcode import cmp_op

import pandas
from matplotlib import pyplot as plt

datasetOriginal = pandas.read_csv('C:\\Users\\macam\\OneDrive\\Escritorio\\UPTC\\ESPECIALIZACIÓN\\4. MODELOS ANALITICOS\\agricultural.csv')

print("DATASET AGRICULTURAL")
datasetOriginal.head()
#%%
# 1. Verificación de consistencia en los datos
print("ANALISIS COLUMNAS DATASET")
datasetOriginal.info()
# El dataset cuenta con 361 registros y 25 columnas, que combinan datos numéricos y categóricos. Se observa que varias columnas presentan valores faltantes (especialmente las relacionadas con precios y porcentajes de cambio), y algunas están registradas como tipo object en lugar de numérico, lo que sugiere la necesidad de limpiar y convertir los datos para garantizar consistencia y permitir análisis precisos.
#%%
# Análisis exploratorio de datos numéricos
print("ANALISIS EXPLORATORIO DATOS NUMERICOS")
datasetOriginal.describe()
# Se obtienen estadísticas descriptivas de las variables numéricas para analizar su rango, dispersión y valores representativos (mínimo, máximo, media y cuartiles).
#%%
# Análisis exploratorio de variables categóricas
print("ANALISIS EXPLORATORIO VARIABLES CATEGORICAS¨")

# Valores únicos de cada columna del dataset
datasetOriginal.apply(pandas.unique)

# Se identifican los valores únicos de cada variable.
# 'Month' presenta valores mensuales de 1990, lo que indica una serie temporal.
# Las demás columnas contienen precios y variaciones porcentuales por producto.
# Los símbolos como '-' o '%' reflejan valores no numéricos que requieren limpieza o conversión.
#%%
# Interpretación de distribuciones de las variables numéricas
print("INTERPRETACION DE LAS DISTRIBUCIONES DE LAS VARIABLES NUMERICA")
import numpy
import matplotlib.pyplot as pyplot
import seaborn as sns
import math
import scipy.stats as stats

# Histogramas
datasetOriginalNumberColumns = datasetOriginal.select_dtypes(include=[numpy.number])
datasetOriginalNumberColumns.info()

pyplot.figure(figsize=(15,10))
datasetOriginalNumberColumns.hist(bins = 30, figsize = (15,15))
pyplot.suptitle("HISTOGRAMAS VARIABLES NUMERICAS")
pyplot.tight_layout()
pyplot.show()

# Diagrama de bigotes
numCols = len(datasetOriginalNumberColumns.columns)
cols = 4
rows  = math.ceil(numCols / cols)
pyplot.figure(figsize=(15,10))

for i, column in enumerate(datasetOriginalNumberColumns, 1):
    pyplot.subplot(rows, cols, i)
    sns.boxplot(data = datasetOriginalNumberColumns[column], color = 'red')
    pyplot.title(column)

pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
pyplot.show()

# QQ Plots
numCols = len(datasetOriginalNumberColumns.columns)
cols = 4
rows  = math.ceil(numCols / cols)
pyplot.figure(figsize=(15,10))

for i, column in enumerate(datasetOriginalNumberColumns, 1):
    pyplot.subplot(rows, cols, i)
    stats.probplot(datasetOriginalNumberColumns[column], plot=pyplot, dist="norm")
    pyplot.title(column)

pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
pyplot.show()

# Se analizan las distribuciones de las variables numéricas mediante histogramas, diagramas de caja (boxplots) y gráficos QQ.
# Los histogramas permiten observar la forma general de la distribución (simétrica o sesgada).
# Los boxplots facilitan la detección de valores atípicos.
# Los QQ plots comparan la distribución de los datos con una normal teórica, ayudando a evaluar si las variables siguen una distribución normal.
#%%
# Interpretación de las distribuciones de las variables categoricas
print("INTERPRETACION DE LAS DISTRIBUCIONES DE LAS VARIABLES CATEGORICAS")

datasetOriginal.apply(pandas.unique)

# Se observan las categorías y valores únicos de cada variable.
# La columna 'Month' muestra una secuencia temporal desde 1990,
# lo que indica una serie cronológica mensual.
# Las variables de precios y cambios porcentuales contienen valores numéricos y símbolos como “%” o “-”,
# lo que sugiere que necesitan limpieza o conversión antes del análisis estadístico.
# Algunas columnas combinan texto y números (por ejemplo “1,071.63” o “-1.35%”),
# lo cual confirma que su tipo de dato actual es 'object' y deben transformarse a valores numéricos.
#%%
# 2. Revisión de datos faltantes
#       a. Datos nulos
datasetOriginal.isnull().sum()

# El análisis evidencia valores faltantes en múltiples columnas numéricas, especialmente en precios y porcentajes de cambio (por ejemplo: Fine wool, Hide, Softlog, etc.). Esto indica que en ciertos meses no se registraron precios o variaciones.
# Las variables sin nulos (como Cotton o Rubber) tienen registros completos y consistentes.
#%%
# Revisión de datos faltantes por porcentaje %
print("Identificacion de valores nulos por porcentaje")
totalRows = len(datasetOriginal)
missingPercentage = datasetOriginal.isnull().sum() / totalRows * 100
print(missingPercentage)
#%%
# Revisión de datos faltantes por gráfico
print("Revision grafica valores nulos")
pyplot.figure(figsize=(15,10))
sns.heatmap(datasetOriginal.isnull(), annot=False, cmap='Blues')
pyplot.show()

# El mapa de calor confirma la presencia de valores nulos en varias columnas, principalmente en las series relacionadas con "wool", "hide", "softlog" y "sawnwood". Esto indica ausencia de registros en algunos periodos, mientras que el resto de variables mantienen datos completos.
#%%
# Revision datos faltantes
#   b. Manejo de datos faltantes

datasetOriginalClean = datasetOriginal.copy()
datasetOriginalClean.info()

# Se crea una copia del dataset original para limpiar los datos sin modificar el original.
# Al revisar la información del nuevo dataset, se observa que varias columnas presentan valores faltantes, especialmente en Coarse wool Price, Fine wool Price, Hard sawnwood Price, Hide Price, Softlog Price, Soft sawnwood Price y Wood pulp Price, junto con sus respectivas variaciones porcentuales.
# Además, varias columnas con tipo 'object' contienen valores numéricos almacenados como texto.
#%%
# Eliminar filas que contengan valores nulos o faltantes
datasetOriginalClean = datasetOriginal.dropna()
#%%
# Cambios en el dataset luego de limpieza
datasetOriginalClean.info()
totalRows = len(datasetOriginal)
totalRowsClean = len(datasetOriginalClean)
totalCleanPercentage = 100 - totalRowsClean / totalRows * 100
print("Diferencia de filas despues de la limpieza del dataset completo: ",  (totalRows - totalRowsClean))
print("Diferencia de filas en porcentaje despues de la limpieza del dataset completo: ",  totalCleanPercentage)
print("Revision grafica valores nulos")
pyplot.figure(figsize=(15,10))
sns.heatmap(datasetOriginalClean.isnull(), annot=False, cmap='Blues')
pyplot.show()

# Tras la limpieza del dataset se eliminaron 34 registros (≈9.4% del total) que contenían valores nulos o inconsistentes. El nuevo conjunto de datos conserva 327 filas con información completa y coherente, sin valores faltantes, lo cual permite continuar con el análisis de forma confiable.
#%%
# Eliminar filas con valores nulos en una columna especifica
datasetOriginalCleanColumns = datasetOriginal.drop(columns = ['Coarse wool Price'])
datasetOriginalCleanColumns = datasetOriginalCleanColumns.dropna(subset=['Fine wool price % Change'])
datasetOriginalCleanColumns.isnull().sum()

#Se eliminaron las filas que contenían valores nulos en la columna “Fine wool price % Change” y se excluyó la columna “Coarse wool Price”. El dataset resultante no presenta valores faltantes en ninguna variable, lo que garantiza una base de datos consistente y lista para el análisis posterior.
#%%
# Verificación final de limpieza:
totalRows = len(datasetOriginal)
totalRowsClean = len(datasetOriginalCleanColumns)
totalCleanPercentage = 100 - totalRowsClean / totalRows * 100
print("Diferencia de filas despues de la limpieza del dataset completo: ", (totalRows - totalRowsClean))
print("Diferencia de filas en porcentaje despues de la limpieza del dataset completo: ", totalCleanPercentage)

pyplot.figure(figsize=(15, 10))
sns.heatmap(datasetOriginalCleanColumns.isnull(), annot=False, cmap='Blues')
pyplot.show()

# Luego del proceso de limpieza se eliminaron 34 registros, equivalentes al 9.42% del total, debido a la presencia de valores nulos en algunas variables. El dataset final no contiene valores faltantes, garantizando así la consistencia y calidad de los datos para los análisis posteriores.
#%%
# Limpieza condicional por columnas:
for col in datasetOriginalCleanColumns.columns:
    null_rows = datasetOriginalCleanColumns[datasetOriginalCleanColumns[col].isna()]
    porcetageDelete = int(len(null_rows[col]) * 0.1)
    if porcetageDelete > 0:
        rowsDelete = null_rows.sample(n=porcetageDelete, random_state=50).index
        datasetOriginalCleanColumns.drop(index=rowsDelete, inplace=True)

datasetOriginalCleanColumns.isnull().sum()

#Tras aplicar una limpieza condicional por columnas, se eliminaron registros con valores nulos de manera controlada (hasta el 10% de los datos faltantes por columna). Como resultado, el dataset quedó completamente libre de valores nulos, garantizando una base de datos íntegra y lista para el análisis estadístico y la modelación.