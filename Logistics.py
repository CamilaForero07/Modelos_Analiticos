#%%
# ANALISIS EXPLORATORIO EDA - LOGISTICS DATASET
from opcode import cmp_op

import pandas
from matplotlib import pyplot as plt

datasetOriginal = pandas.read_csv('C:\\Users\\macam\\OneDrive\\Escritorio\\UPTC\\ESPECIALIZACIÓN\\4. MODELOS ANALITICOS\\logistics_dataset.csv')

print("LOGISTICS DATASET")
datasetOriginal.head()
#%%
# Verificación de consistencia en los datos
print("ANALISIS COLUMNAS DATASET")
datasetOriginal.info()

# Análisis de consistencia del dataset:
#El dataset contiene 3.204 registros y 23 columnas. Todas las variables tienen datos completos, sin valores nulos. Los tipos de datos son correctos: numéricos (int64, float64) y categóricos (object). La información es consistente y está lista para el análisis exploratorio.
#%%
# Análisis exploratorio de datos numéricos
print("ANALISIS EXPLORATORIO DATOS NUMERICOS")
datasetOriginal.describe()
#%%
# Análisis exploratorio de variables categóricas
print("ANALISIS EXPLORATORIO VARIABLES CATEGORICAS¨")

# Valores únicos de cada columna del dataset
datasetOriginal.apply(pandas.unique)
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
#%%
#Interpretación de las distribuciones de las variables categoricas
print("INTERPRETACION DE LAS DISTRIBUCIONES DE LAS VARIABLES CATEGORICAS")

datasetOriginal.apply(pandas.unique)

#Interpretación de las distribuciones de las variables categóricas:
#Se identificaron las variables categóricas del dataset y sus posibles usos analíticos.
# Las variables category y zone presentan pocas categorías, por lo que son útiles para comparar el rendimiento logístico entre grupos de productos o zonas operativas.
# Variables como item_id y storage_location_id tienen valores únicos que las identifican, pero no aportan valor analítico directo.
# Finalmente, last_restock_date evidencia fechas de reposición distribuidas durante todo el año 2024, reflejando un proceso de abastecimiento constante.
#%%
# Eliminando columnas numéricas
datasetOriginalCategoryColumns = datasetOriginal.drop(columns = ['stock_level','reorder_point','reorder_frequency_days','lead_time_days','daily_demand','demand_std_dev','item_popularity_score','picking_time_seconds','handling_cost_per_unit','unit_price','holding_cost_per_unit_day','stockout_count_last_month','order_fulfillment_rate','total_orders_last_month','turnover_ratio','layout_efficiency_score','forecasted_demand_next_7d','KPI_score'])

# Variable categórica 'Category'
pyplot.figure(figsize=(15,10))
sns.countplot(y = datasetOriginal['category'], data = datasetOriginal, order=datasetOriginal['category'].value_counts().index, hue='category', dodge=False)
pyplot.title('category')
pyplot.xlabel('category')
pyplot.ylabel('Count')
pyplot.show()

# Variable categórica 'Zone'
pyplot.figure(figsize=(15,10))
sns.countplot(y = datasetOriginal['zone'], data = datasetOriginal, order=datasetOriginal['zone'].value_counts().index, hue='zone', dodge=False)
pyplot.title('zone')
pyplot.xlabel('zone')
pyplot.ylabel('Count')
pyplot.show()
#%%
# Contar el numero de similitudes entre 2 columnas
cross_table = pandas.crosstab(datasetOriginalCategoryColumns['category'], datasetOriginalCategoryColumns['zone'])
pyplot.figure(figsize=(15,10))
sns.heatmap(cross_table, annot=True, fmt='d', cmap='viridis')
pyplot.title('Relación entre category y zone')
pyplot.xlabel('category')
pyplot.ylabel('zone')
pyplot.show()
#%%
# Revisión de datos faltantes
datasetOriginal.isnull().sum()

#NO SE PRESENTAN VALORES NULOS
#%%
# Revision datos faltantes
#   Manejo de datos faltantes

datasetOriginalClean = datasetOriginal.copy()
datasetOriginalClean.info()