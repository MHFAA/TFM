#!/usr/bin/env python
# coding: utf-8

# ## Importamos las librerías necesarias🗂️

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import warnings
warnings.filterwarnings('ignore')
# Imprimimos la ruta de trabajo
print(os.getcwd())


# ## Creación de los directorios 📂

# In[21]:


# Creamos los directorios de trabajo, sobre los que guardaremos los datos
# Carpeta processed_data y sus subcarpetas
if not os.path.exists('./processed_data'):
    os.makedirs('./processed_data')
# Carpeta biorreactores y centrifugadoras dentro de processed_data
if not os.path.exists('./processed_data/biorreactores'):
    os.makedirs('./processed_data/biorreactores')
if not os.path.exists('./processed_data/centrifugadoras'):
    os.makedirs('./processed_data/centrifugadoras')
# Carpeta train y test dentro de processed_data
if not os.path.exists('./processed_data/train'):
    os.makedirs('./processed_data/train')
if not os.path.exists('./processed_data/test'):
    os.makedirs('./processed_data/test')


# 
# ### Procesamiento las OF 
# 
# 
# Las ofs que seleccionemos son las que usaremos para el entrenamiento de nuestro modelo. 
# 

# In[22]:


of = pd.read_excel('data/OF 123456 v03.xlsx')
# Eliminamos las columnas Unidad de medida, Número material y Texto breve material
of = of.drop(columns=['Unidad de medida', 'Número material', 'Texto breve material','Orden'])
# Pasamos la columna Cantidad entregada a float
of['Cantidad entregada'] = of['Cantidad entregada'].astype(float)
# SI hay alguna cantidad entregada igual o menor a 1, la cambiamos por la media de las cantidades entregadas
mean = of['Cantidad entregada'].mean()
of['Cantidad entregada'] = of['Cantidad entregada'].apply(lambda x: mean if x <= 1 else x)
# Eliminamos la fila que tiene un el valor Lote a P23273
#of = of[of['Lote'] != 'P23273']
# Modificamos Lote para que deje de ser 23/019 y sea 23019 en formato int
of['Lote'] = of['Lote'].apply(lambda x: str(x.replace('/', '')))
of = of.rename(columns={'Cantidad entregada': 'CantidadEntregada'})
# Nos guardamos en un txt los lotes que hay en of
lotes = of['Lote']
with open('data/lotes.txt', 'w') as f:
    for lote in lotes:
        f.write(str(lote) + '\n')
of.to_csv('./processed_data/of.csv', index=False)


# ###  Procesamiento del Preinóculo 🧫

# In[23]:


preinoculo = pd.read_excel('./data/Fases produccion v03.xlsx', sheet_name='Preinóculo')
print("El número de filas y columnas de preinoculo es: ", preinoculo.shape)
# Cambiamos el nombre de la columna Unnamed: 0 a Lote, Unnamed: 1 a F_h_init, Unnamed: 2 a F_h_end
preinoculo = preinoculo.rename(columns={'Unnamed: 0': 'Lote', 'Unnamed: 1': 'F_h_init_prein', 'Unnamed: 2': 'F_h_end_prein'})
# Siguiendo con los cambios, cambiamos el nombre de la columna pH a pH_1, Unnamed: 4 a pH_2, Unnamed: 5 a pH_3
preinoculo = preinoculo.rename(columns={'pH': 'pH_1', 'Unnamed: 4': 'pH_2', 'Unnamed: 5': 'pH_3'})
# Haremos lo mismo con la columna Turbidez que pasará a ser Turbidez_1, Unnamed: 7 a Turbidez_2, Unnamed: 8 a Turbidez_3
preinoculo = preinoculo.rename(columns={'Turbidez': 'Turbidez_1', 'Unnamed: 7': 'Turbidez_2', 'Unnamed: 8': 'Turbidez_3'})
# Finalmente cambiaremos el nombre de la columna Línea utilizada a Linea_utilizada_1, Unnamed: 10 a Linea_utilizada_2, Unnamed: 11 a Linea_utilizada_3
preinoculo = preinoculo.rename(columns={'Línea utilizada': 'Linea_utilizada_1', 'Unnamed: 10': 'Linea_utilizada_2', 'Unnamed: 11': 'Linea_utilizada_3'})
# Ahora nos cargamos la fila 0
preinoculo = preinoculo.drop(0)
# Pasamos el lote a Int
try:
    preinoculo['Lote'] = preinoculo['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    preinoculo = preinoculo[preinoculo['Lote'] != 'Lote']
print("El número de filas y columnas de preinoculo es: ", preinoculo.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en preinoculo
preinoculo = preinoculo[preinoculo['Lote'].isin(of['Lote'])]
print("El número de filas y columnas de preinoculo es: ", preinoculo.shape)
# Comprobamos que la suma de las columnas línea 1, línea 2 y línea 3 es igual a 2
# En caso de que no, mostramos que filas no cumplen con la condición
preinoculo['Linea_utilizada_1'] = preinoculo['Linea_utilizada_1'].astype(int)
preinoculo['Linea_utilizada_2'] = preinoculo['Linea_utilizada_2'].astype(int)
preinoculo['Linea_utilizada_3'] = preinoculo['Linea_utilizada_3'].astype(int)
# Creamos una nueva columna que se llame duracion_preinoculo que será la resta de F_h_end_prein - F_h_init_prein
preinoculo['F_h_init_prein'] = pd.to_datetime(preinoculo['F_h_init_prein'], errors='coerce')
preinoculo['F_h_end_prein'] = pd.to_datetime(preinoculo['F_h_end_prein'], errors='coerce')

# Añadimos una columna de duracion de la fase de preinoculo
preinoculo['Duracion_preinoculo'] = preinoculo['F_h_end_prein'] - preinoculo['F_h_init_prein']
# Si dura más de 10 dia, le restamos 10 día
preinoculo['Duracion_preinoculo'] = preinoculo['Duracion_preinoculo'].apply(lambda x: x - pd.Timedelta(days=10) if x.days > 10 else x)
# SI hay alguna diracion negativa, la cambiamos a la media de las duraciones
preinoculo['Duracion_preinoculo'] = preinoculo['Duracion_preinoculo'].apply(lambda x: x if x.total_seconds() > 0 else preinoculo['Duracion_preinoculo'].mean())
# Eliminamos las columnas F_h_init_prein y F_h_end_prein
#preinoculo = preinoculo.drop(columns=['F_h_init_prein', 'F_h_end_prein'])

# Pasamos las columnas pH_1, pH_2 y pH_3 a float
preinoculo['pH_1'] = pd.to_numeric(preinoculo['pH_1'], errors='coerce')
preinoculo['pH_1'] = preinoculo['pH_1'].astype(float)
preinoculo['pH_2'] = pd.to_numeric(preinoculo['pH_2'], errors='coerce')
preinoculo['pH_2'] = preinoculo['pH_2'].astype(float)
# En pH_3 hay un valor que no es un número, lo cambiamos a NaN
preinoculo['pH_3'] = pd.to_numeric(preinoculo['pH_3'], errors='coerce')
preinoculo['pH_3'] = preinoculo['pH_3'].astype(float)
# Pasamos las columnas Turbidez_1, Turbidez_2 y Turbidez_3 a float
preinoculo['Turbidez_1'] = preinoculo['Turbidez_1'].astype(float)
preinoculo['Turbidez_2'] = preinoculo['Turbidez_2'].astype(float)
# En Turbidez_3 hay un valor que no es un número, lo cambiamos a NaN
preinoculo['Turbidez_3'] = pd.to_numeric(preinoculo['Turbidez_3'], errors='coerce')
preinoculo['Turbidez_3'] = preinoculo['Turbidez_3'].astype(float)

# Para esas filas con nulos, lo que haremos será rellenar los valores nulos con la media de los valores de la columna
preinoculo['pH_1'] = preinoculo['pH_1'].fillna(preinoculo['pH_1'].mean())
preinoculo['pH_2'] = preinoculo['pH_2'].fillna(preinoculo['pH_2'].mean())
preinoculo['pH_3'] = preinoculo['pH_3'].fillna(preinoculo['pH_3'].mean())
preinoculo['Turbidez_1'] = preinoculo['Turbidez_1'].fillna(preinoculo['Turbidez_1'].mean())
preinoculo['Turbidez_2'] = preinoculo['Turbidez_2'].fillna(preinoculo['Turbidez_2'].mean())
preinoculo['Turbidez_3'] = preinoculo['Turbidez_3'].fillna(preinoculo['Turbidez_3'].mean())
preinoculo['Linea_utilizada_1'] = preinoculo['Linea_utilizada_1'].astype(int)
preinoculo['Linea_utilizada_2'] = preinoculo['Linea_utilizada_2'].astype(int)
preinoculo['Linea_utilizada_3'] = preinoculo['Linea_utilizada_3'].astype(int)
# Ahora solo tendremos dos columnas de ph y turbidez, ya que solo imputaremos las mediciones cuyas lineas hayan sido utilizadas (que tengan un valor distinto de 0)
# Creamos las columnas pH_1_utilizada, pH_2_utilizada, Turbidez_1_utilizada y Turbidez_2_utilizada
preinoculo['pH_1_utilizada'] = 0
preinoculo['pH_1_utilizada'] = preinoculo['pH_1_utilizada'].astype(float)
preinoculo['pH_2_utilizada'] = 0 
preinoculo['pH_2_utilizada'] = preinoculo['pH_2_utilizada'].astype(float)
preinoculo['Turbidez_1_utilizada'] = 0
preinoculo['Turbidez_1_utilizada'] = preinoculo['Turbidez_1_utilizada'].astype(float)
preinoculo['Turbidez_2_utilizada'] = 0
preinoculo['Turbidez_2_utilizada'] = preinoculo['Turbidez_2_utilizada'].astype(float)
# recorremos las filas
for index, row in preinoculo.iterrows():
    if row['Linea_utilizada_1'] != 0:
        preinoculo.at[index, 'pH_1_utilizada'] = row['pH_1']
        preinoculo.at[index, 'Turbidez_1_utilizada'] = row['Turbidez_1']
    else:
        preinoculo.at[index, 'pH_1_utilizada'] = row['pH_3']
        preinoculo.at[index, 'Turbidez_1_utilizada'] = row['Turbidez_3']
    if row['Linea_utilizada_2'] != 0:
        preinoculo.at[index, 'pH_2_utilizada'] = row['pH_2']
        preinoculo.at[index, 'Turbidez_2_utilizada'] = row['Turbidez_2']
    else:
        preinoculo.at[index, 'pH_2_utilizada'] = row['pH_3']
        preinoculo.at[index, 'Turbidez_2_utilizada'] = row['Turbidez_3']
# Hacemos drop de las columnas que ya no necesitamos
preinoculo = preinoculo.drop(columns=['pH_1', 'pH_2', 'pH_3', 'Turbidez_1', 'Turbidez_2', 'Turbidez_3'])
preinoculo = preinoculo.drop(columns=['Linea_utilizada_1', 'Linea_utilizada_2', 'Linea_utilizada_3'])
# Nos guardamos el preinoculo
# Antes de guardarlo, tenemos 2 lotes 24020, nos cargamos solo uno y dejamos el otro
preinoculo = preinoculo.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de preinoculo es: ", preinoculo.shape)
preinoculo.to_csv('./processed_data/preinoculo.csv', index=False)


# ###  Procesamiento del inóculo 🧪

# In[24]:


inoculo = pd.read_excel('./data/Fases produccion v03.xlsx', sheet_name='Inóculo')
inoculo = inoculo.rename(columns={'LOTE': 'Lote', 'ID bioreactor': 'ID_bioreactor', 'Fecha/hora inicio': 'F_h_init_in', 'Fecha/hora fin': 'F_h_end_in','Volumen de cultivo': 'Volumen_cultivo', 'Turbidez inicio cultivo': 'Turbidez_init', 'Turbidez final culttivo': 'Turbidez_end', 'Viabilidad final cultivo': 'Vialidad_end'})
print("El número de filas y columnas de inoculo es: ", inoculo.shape)
try:
    inoculo['Lote'] = inoculo['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    inoculo = inoculo[inoculo['Lote'] != 'Lote']
print("El número de filas y columnas de inoculo es: ", inoculo.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en inoculo
inoculo = inoculo[inoculo['Lote'].isin(of['Lote'])]
inoculo = inoculo.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de inoculo es: ", inoculo.shape)
# Pasamos la columna F_h_init_in y F_h_end_in a datetime
inoculo['F_h_init_in'] = pd.to_datetime(inoculo['F_h_init_in'], format='%d.%m.%Y %H:%M:%S')
inoculo['F_h_end_in'] = pd.to_datetime(inoculo['F_h_end_in'], format='%d.%m.%Y %H:%M:%S')
# Añadimos una columna de duracion de la fase de inóculo
inoculo['Duracion_inoculo'] = inoculo['F_h_end_in'] - inoculo['F_h_init_in']
# SI hay alguna diracion negativa, la cambiamos a la media de las duraciones
inoculo['Duracion_inoculo'] = inoculo['Duracion_inoculo'].apply(lambda x: x if x.total_seconds() > 0 else inoculo['Duracion_inoculo'].mean())
# Pasamos la columna ID_bioreactor a int
inoculo['ID_bioreactor'] = inoculo['ID_bioreactor'].astype(int)
# Volumen_cultivo, Turbidez_init, Turbidez_end y Vialidad_end a float, pero antes, si tienen un NaN, lo cambiamos a pd.to_numeric
inoculo['Volumen_cultivo'] = pd.to_numeric(inoculo['Volumen_cultivo'], errors='coerce')
inoculo['Volumen_cultivo'] = inoculo['Volumen_cultivo'].astype(float)
# Si hay algún valor menor o igual a 0, lo cambiamos a la media de los valores
mean = inoculo['Volumen_cultivo'].mean()
inoculo['Volumen_cultivo'] = inoculo['Volumen_cultivo'].apply(lambda x: mean if x <= 0 else x)
inoculo['Turbidez_init'] = pd.to_numeric(inoculo['Turbidez_init'], errors='coerce')
inoculo['Turbidez_init'] = inoculo['Turbidez_init'].astype(float)
mean = inoculo['Turbidez_init'].mean()
inoculo['Turbidez_init'] = inoculo['Turbidez_init'].apply(lambda x: mean if x <= 0 else x)
inoculo['Turbidez_end'] = pd.to_numeric(inoculo['Turbidez_end'], errors='coerce')
inoculo['Turbidez_end'] = inoculo['Turbidez_end'].astype(float)
mean = inoculo['Turbidez_end'].mean()
inoculo['Turbidez_end'] = inoculo['Turbidez_end'].apply(lambda x: mean if x <= 0 else x)
inoculo['Vialidad_end'] = pd.to_numeric(inoculo['Vialidad_end'], errors='coerce')
inoculo['Vialidad_end'] = inoculo['Vialidad_end'].astype(float)
mean = inoculo['Vialidad_end'].mean()
inoculo['Vialidad_end'] = inoculo['Vialidad_end'].apply(lambda x: mean if x <= 0 else x)
# Nos guardamos el inoculo
inoculo.to_csv('./processed_data/inoculo.csv', index=False)


# ###  Procesamiento del cultivo 🌱

# In[25]:


cultivo = pd.read_excel('./data/Fases produccion v03.xlsx', sheet_name='Cultivo final')
cultivo = cultivo.rename(columns={'LOTE': 'Lote','Orden en el encadenado':'orden_encadenado', 'ID Bioreactor': 'ID_bioreactor', 'LOTE parental': 'Lote_parental', 'Fecha/hora inicio': 'F_h_init_cul', 'Fecha/hora fin': 'F_h_end_cul', 'Volumen de inóculo utilizado': 'Volumen_inoculo_used', 'Turbidez inicio cultivo': 'Turbidez_init_cul', 'Turbidez fin cultivo': 'Turbidez_end_cul', 'Viabilidad final cultivo': 'Vialidad_end_cul', 'ID Centrífuga': 'ID_centrifuga', 'Centrifugación 1 turbidez': 'Centrifugacion_1_turbidez', 'Centrifugación 2 turbidez': 'Centrifugacion_2_turbidez'})
# Pasamos la columna F_h_init_cul y F_h_end_cul a datetime
cultivo['F_h_init_cul'] = pd.to_datetime(cultivo['F_h_init_cul'], format='%d.%m.%Y %H:%M:%S')
cultivo['F_h_end_cul'] = pd.to_datetime(cultivo['F_h_end_cul'], format='%d.%m.%Y %H:%M:%S')
# Añadimos una columna de duracion de la fase de cultivo
cultivo['Duracion_cultivo'] = cultivo['F_h_end_cul'] - cultivo['F_h_init_cul']
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
try:
    cultivo['Lote'] = cultivo['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    cultivo = cultivo[cultivo['Lote'] != 'Lote']
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en cultivo
cultivo = cultivo[cultivo['Lote'].isin(of['Lote'])]
cultivo = cultivo.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
# Pasamos la columna Volumen_inoculo_used a float
cultivo['Volumen_inoculo_used'] = pd.to_numeric(cultivo['Volumen_inoculo_used'], errors='coerce')
cultivo['Volumen_inoculo_used'] = cultivo['Volumen_inoculo_used'].astype(float)
mean = cultivo['Volumen_inoculo_used'].mean()
cultivo['Volumen_inoculo_used'] = cultivo['Volumen_inoculo_used'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna Turbidez_init_cul y Turbidez_end_cul a float
cultivo['Turbidez_init_cul'] = pd.to_numeric(cultivo['Turbidez_init_cul'], errors='coerce')
cultivo['Turbidez_init_cul'] = cultivo['Turbidez_init_cul'].astype(float)
mean = cultivo['Turbidez_init_cul'].mean()
cultivo['Turbidez_init_cul'] = cultivo['Turbidez_init_cul'].apply(lambda x: mean if x <= 0 else x)
cultivo['Turbidez_end_cul'] = pd.to_numeric(cultivo['Turbidez_end_cul'], errors='coerce')
cultivo['Turbidez_end_cul'] = cultivo['Turbidez_end_cul'].astype(float)
mean = cultivo['Turbidez_end_cul'].mean()
cultivo['Turbidez_end_cul'] = cultivo['Turbidez_end_cul'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna Vialidad_end_cul a float
cultivo['Vialidad_end_cul'] = pd.to_numeric(cultivo['Vialidad_end_cul'], errors='coerce')
cultivo['Vialidad_end_cul'] = cultivo['Vialidad_end_cul'].astype(float)
mean = cultivo['Vialidad_end_cul'].mean()
cultivo['Vialidad_end_cul'] = cultivo['Vialidad_end_cul'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna ID_centrifuga a int
cultivo['ID_centrifuga'] = cultivo['ID_centrifuga'].astype(int)
# Pasamos la columna Centrifugacion_1_turbidez y Centrifugacion_2_turbidez a float
cultivo['Centrifugacion_1_turbidez'] = pd.to_numeric(cultivo['Centrifugacion_1_turbidez'], errors='coerce')
cultivo['Centrifugacion_1_turbidez'] = cultivo['Centrifugacion_1_turbidez'].astype(float)
mean = cultivo['Centrifugacion_1_turbidez'].mean()
cultivo['Centrifugacion_1_turbidez'] = cultivo['Centrifugacion_1_turbidez'].apply(lambda x: mean if x <= 0 else x)
cultivo['Centrifugacion_2_turbidez'] = pd.to_numeric(cultivo['Centrifugacion_2_turbidez'], errors='coerce')
cultivo['Centrifugacion_2_turbidez'] = cultivo['Centrifugacion_2_turbidez'].astype(float)
mean = cultivo['Centrifugacion_2_turbidez'].mean()
cultivo['Centrifugacion_2_turbidez'] = cultivo['Centrifugacion_2_turbidez'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna Producto 1 a Float
cultivo['Producto 1'] = pd.to_numeric(cultivo['Producto 1'], errors='coerce')
cultivo['Producto 1'] = cultivo['Producto 1'].astype(float)
mean = cultivo['Producto 1'].mean()
cultivo['Producto 1'] = cultivo['Producto 1'].apply(lambda x: mean if x <= 0 else x)
# Nos guardamos el cultivo
cultivo.to_csv('./processed_data/cultivo.csv', index=False)


# In[26]:


cultivo = pd.read_excel('./data/Fases produccion v03 Test.xlsx', sheet_name='Cultivo final')
cultivo = cultivo.rename(columns={'LOTE': 'Lote','Orden en el encadenado':'orden_encadenado', 'ID Bioreactor': 'ID_bioreactor', 'LOTE parental': 'Lote_parental', 'Fecha/hora inicio': 'F_h_init_cul', 'Fecha/hora fin': 'F_h_end_cul', 'Volumen de inóculo utilizado': 'Volumen_inoculo_used', 'Turbidez inicio cultivo': 'Turbidez_init_cul', 'Turbidez fin cultivo': 'Turbidez_end_cul', 'Viabilidad final cultivo': 'Vialidad_end_cul', 'ID Centrífuga': 'ID_centrifuga', 'Centrifugación 1 turbidez': 'Centrifugacion_1_turbidez', 'Centrifugación 2 turbidez': 'Centrifugacion_2_turbidez'})
# Pasamos la columna F_h_init_cul y F_h_end_cul a datetime
cultivo['F_h_init_cul'] = pd.to_datetime(cultivo['F_h_init_cul'], format='%d.%m.%Y %H:%M:%S')
cultivo['F_h_end_cul'] = pd.to_datetime(cultivo['F_h_end_cul'], format='%d.%m.%Y %H:%M:%S')
# Añadimos una columna de duracion de la fase de cultivo
cultivo['Duracion_cultivo'] = cultivo['F_h_end_cul'] - cultivo['F_h_init_cul']
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
try:
    cultivo['Lote'] = cultivo['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    cultivo = cultivo[cultivo['Lote'] != 'Lote']
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en cultivo
cultivo = cultivo[cultivo['Lote'].isin(of['Lote'])]
cultivo = cultivo.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de cultivo es: ", cultivo.shape)
# Pasamos la columna Volumen_inoculo_used a float
cultivo['Volumen_inoculo_used'] = pd.to_numeric(cultivo['Volumen_inoculo_used'], errors='coerce')
cultivo['Volumen_inoculo_used'] = cultivo['Volumen_inoculo_used'].astype(float)
mean = cultivo['Volumen_inoculo_used'].mean()
cultivo['Volumen_inoculo_used'] = cultivo['Volumen_inoculo_used'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna Turbidez_init_cul y Turbidez_end_cul a float
cultivo['Turbidez_init_cul'] = pd.to_numeric(cultivo['Turbidez_init_cul'], errors='coerce')
cultivo['Turbidez_init_cul'] = cultivo['Turbidez_init_cul'].astype(float)
mean = cultivo['Turbidez_init_cul'].mean()
cultivo['Turbidez_init_cul'] = cultivo['Turbidez_init_cul'].apply(lambda x: mean if x <= 0 else x)
cultivo['Turbidez_end_cul'] = pd.to_numeric(cultivo['Turbidez_end_cul'], errors='coerce')
cultivo['Turbidez_end_cul'] = cultivo['Turbidez_end_cul'].astype(float)
mean = cultivo['Turbidez_end_cul'].mean()
cultivo['Turbidez_end_cul'] = cultivo['Turbidez_end_cul'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna Vialidad_end_cul a float
cultivo['Vialidad_end_cul'] = pd.to_numeric(cultivo['Vialidad_end_cul'], errors='coerce')
cultivo['Vialidad_end_cul'] = cultivo['Vialidad_end_cul'].astype(float)
mean = cultivo['Vialidad_end_cul'].mean()
cultivo['Vialidad_end_cul'] = cultivo['Vialidad_end_cul'].apply(lambda x: mean if x <= 0 else x)
# Pasamos la columna ID_centrifuga a int
cultivo['ID_centrifuga'] = cultivo['ID_centrifuga'].astype(int)
# Pasamos la columna Centrifugacion_1_turbidez y Centrifugacion_2_turbidez a float
cultivo['Centrifugacion_1_turbidez'] = pd.to_numeric(cultivo['Centrifugacion_1_turbidez'], errors='coerce')
cultivo['Centrifugacion_1_turbidez'] = cultivo['Centrifugacion_1_turbidez'].astype(float)
mean = cultivo['Centrifugacion_1_turbidez'].mean()
cultivo['Centrifugacion_1_turbidez'] = cultivo['Centrifugacion_1_turbidez'].apply(lambda x: mean if x <= 0 else x)
cultivo['Centrifugacion_2_turbidez'] = pd.to_numeric(cultivo['Centrifugacion_2_turbidez'], errors='coerce')
cultivo['Centrifugacion_2_turbidez'] = cultivo['Centrifugacion_2_turbidez'].astype(float)
mean = cultivo['Centrifugacion_2_turbidez'].mean()
cultivo['Centrifugacion_2_turbidez'] = cultivo['Centrifugacion_2_turbidez'].apply(lambda x: mean if x <= 0 else x)
# Nos guardamos el cultivo
cultivo.to_csv('./processed_data/cultivo_test.csv', index=False)


# ### Procesamos los Cinéticos 📈

# In[27]:


# Ahora leemos Cinéticos IPC.xlsx, pero solo la página de Inóculos
cineticos_inoculos = pd.read_excel('data/Cineticos IPC.xlsx', sheet_name='Inóculos')
print("El número de filas y columnas de cineticos_inoculos es: ", cineticos_inoculos.shape)
try:
    cineticos_inoculos['Lote'] = cineticos_inoculos['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    cineticos_inoculos = cineticos_inoculos[cineticos_inoculos['Lote'] != 'Lote']
print("El número de filas y columnas de cineticos_inoculos es: ", cineticos_inoculos.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en cineticos_inoculos
cineticos_inoculos = cineticos_inoculos[cineticos_inoculos['Lote'].isin(of['Lote'])]
print("El número de filas y columnas de cineticos_inoculos es: ", cineticos_inoculos.shape)
# Pasamos la columna Fecha de inicio a datetime con horas y minutos
cineticos_inoculos['Fecha'] = pd.to_datetime(cineticos_inoculos['Fecha'], format='%d.%m.%Y %H:%M:%S')
# Cambiamos el nombre de Fecha a F_h_cin_in
cineticos_inoculos = cineticos_inoculos.rename(columns={'Fecha': 'F_h_cin_cul'})
# Pasamos las columnas Turbidez, Viabilidad y Glucosa g/L a float
cineticos_inoculos['Turbidez'] = pd.to_numeric(cineticos_inoculos['Turbidez'], errors='coerce')
cineticos_inoculos['Turbidez'] = cineticos_inoculos['Turbidez'].astype(float)
# Si hay algún valor NaN en Turbidez o Viabilidad, lo cambiamos usando forward fill
cineticos_inoculos['Turbidez'] = cineticos_inoculos['Turbidez'].fillna(method='ffill')
cineticos_inoculos['Viabilidad'] = pd.to_numeric(cineticos_inoculos['Viabilidad'], errors='coerce')
cineticos_inoculos['Viabilidad'] = cineticos_inoculos['Viabilidad'].astype(float)
cineticos_inoculos['Viabilidad'] = cineticos_inoculos['Viabilidad'].fillna(method='ffill')
# Para cada lote sacamos la media de la turbidez y la viabilidad
lotes = cineticos_inoculos['Lote'].unique()
turbideces_min = []
turbideces_mean = []
turbideces_max = []
viabilidades_min = []
viabilidades_mean = []
viabilidades_max = []
for lote in lotes:
    cinetico = cineticos_inoculos[cineticos_inoculos['Lote'] == lote]
    turbideces_min.append(cinetico['Turbidez'].min())
    turbideces_mean.append(cinetico['Turbidez'].mean())
    turbideces_max.append(cinetico['Turbidez'].max())
    viabilidades_min.append(cinetico['Viabilidad'].min())
    viabilidades_mean.append(cinetico['Viabilidad'].mean())
    viabilidades_max.append(cinetico['Viabilidad'].max())
# Creamos un dataframe con los valores
cineticos_inoculos = pd.DataFrame({'Lote': lotes, 'Turbidez_min': turbideces_min, 'Turbidez_mean': turbideces_mean, 'Turbidez_max': turbideces_max, 'Viabilidad_min': viabilidades_min, 'Viabilidad_mean': viabilidades_mean, 'Viabilidad_max': viabilidades_max})
print("El número de filas y columnas de cineticos_inoculos es: ", cineticos_inoculos.shape)
# Nos guardamos el cineticos_inoculos
cineticos_inoculos.to_csv('./processed_data/cineticos_inoculos.csv', index=False)


# ###  Procesamiento de más cinéticos 📈

# In[28]:


cineticos_cultivos = pd.read_excel('data/Cineticos IPC.xlsx', sheet_name='Cultivos finales')
print("El número de filas y columnas de cineticos_cultivos es: ", cineticos_cultivos.shape)
try:
    cineticos_cultivos['Lote'] = cineticos_cultivos['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    cineticos_cultivos = cineticos_cultivos[cineticos_cultivos['Lote'] != 'Lote']
print("El número de filas y columnas de cineticos_cultivos es: ", cineticos_cultivos.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en cineticos_cultivos
cineticos_cultivos = cineticos_cultivos[cineticos_cultivos['Lote'].isin(of['Lote'])]
print("El número de filas y columnas de cineticos_cultivos es: ", cineticos_cultivos.shape)

# Pasamos la columna Fecha de inicio a datetime con horas y minutos
cineticos_cultivos['Fecha'] = pd.to_datetime(cineticos_cultivos['Fecha'], format='%d.%m.%Y %H:%M:%S')
# Cambiamos el nombre de Fecha a F_h_cin_in
cineticos_cultivos = cineticos_cultivos.rename(columns={'Fecha': 'F_h_cin_cul'})
# Pasamos las columnas Turbidez, Viabilidad y Glucosa g/L a float
cineticos_cultivos['Turbidez'] = pd.to_numeric(cineticos_cultivos['Turbidez'], errors='coerce')
cineticos_cultivos['Turbidez'] = cineticos_cultivos['Turbidez'].astype(float)
# Si hay algún valor NaN en Turbidez o Viabilidad, lo cambiamos usando forward fill
cineticos_cultivos['Turbidez'] = cineticos_cultivos['Turbidez'].fillna(method='ffill')
cineticos_cultivos['Viabilidad'] = pd.to_numeric(cineticos_cultivos['Viabilidad'], errors='coerce')
cineticos_cultivos['Viabilidad'] = cineticos_cultivos['Viabilidad'].astype(float)
cineticos_cultivos['Viabilidad'] = cineticos_cultivos['Viabilidad'].fillna(method='ffill')
cineticos_cultivos['Glucosa g/L'] = pd.to_numeric(cineticos_cultivos['Glucosa g/L'], errors='coerce')
cineticos_cultivos['Glucosa g/L'] = cineticos_cultivos['Glucosa g/L'].astype(float)
cineticos_cultivos['Glucosa g/L'] = cineticos_cultivos['Glucosa g/L'].fillna(method='ffill')
# Para cada lote sacamos el minimo, la media y el maximo de la turbidez, la viabilidad y la glucosa
lotes = cineticos_cultivos['Lote'].unique()
turbideces_min = []
turbideces_mean = []
turbideces_max = []
viabilidades_min = []
viabilidades_mean = []
viabilidades_max = []
glucosas_min = []
glucosas_mean = []
glucosas_max = []
for lote in lotes:
    cinetico = cineticos_cultivos[cineticos_cultivos['Lote'] == lote]
    turbideces_min.append(cinetico['Turbidez'].min())
    turbideces_mean.append(cinetico['Turbidez'].mean())
    turbideces_max.append(cinetico['Turbidez'].max())
    viabilidades_min.append(cinetico['Viabilidad'].min())
    viabilidades_mean.append(cinetico['Viabilidad'].mean())
    viabilidades_max.append(cinetico['Viabilidad'].max())
    glucosas_min.append(cinetico['Glucosa g/L'].min())
    glucosas_mean.append(cinetico['Glucosa g/L'].mean())
    glucosas_max.append(cinetico['Glucosa g/L'].max())
# Creamos un dataframe con los valores
cineticos_cultivos = pd.DataFrame({'Lote': lotes, 'Turbidez_min': turbideces_min, 'Turbidez_mean': turbideces_mean, 'Turbidez_max': turbideces_max, 'Viabilidad_min': viabilidades_min, 'Viabilidad_mean': viabilidades_mean, 'Viabilidad_max': viabilidades_max, 'Glucosa_min': glucosas_min, 'Glucosa_mean': glucosas_mean, 'Glucosa_max': glucosas_max})
cineticos_cultivos = cineticos_cultivos.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de cineticos_cultivos es: ", cineticos_cultivos.shape)
# Nos guardamos el cineticos_cultivos
cineticos_cultivos.to_csv('./processed_data/cineticos_cultivos.csv', index=False)


# ###  Procesamiento de más cinéticos 📈

# In[29]:


cineticos_centrifugacion = pd.read_excel('data/Cineticos IPC.xlsx', sheet_name='Centrifugación')
print("El número de filas y columnas de cineticos_centrifugacion es: ", cineticos_centrifugacion.shape)
try:
    cineticos_centrifugacion['Lote'] = cineticos_centrifugacion['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    cineticos_centrifugacion = cineticos_centrifugacion[cineticos_centrifugacion['Lote'] != 'Lote']
print("El número de filas y columnas de cineticos_centrifugacion es: ", cineticos_centrifugacion.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en cineticos_centrifugacion
cineticos_centrifugacion = cineticos_centrifugacion[cineticos_centrifugacion['Lote'].isin(of['Lote'])]
print("El número de filas y columnas de cineticos_centrifugacion es: ", cineticos_centrifugacion.shape)
# La columna Centrifugada (1 o 2) la vamos a eliminar, para crear Volumen_centrifugado_1 y Volumen_centrifugado_2 y Turbidez_centrifugado_1 y Turbidez_centrifugado_2
vol_centrifugado_1 = []
vol_centrifugado_2 = []
turbidez_centrifugado_1 = []
turbidez_centrifugado_2 = []
for index, row in cineticos_centrifugacion.iterrows():
    if row['Centrifugada (1 o 2)'] == 1:
        vol_centrifugado_1.append(row['Volumen centrifugado (L)'])
        turbidez_centrifugado_1.append(row['Turbidez'])
        vol_centrifugado_2.append(0)
        turbidez_centrifugado_2.append(0)
    else:
        vol_centrifugado_2.append(row['Volumen centrifugado (L)'])
        turbidez_centrifugado_2.append(row['Turbidez'])
        vol_centrifugado_1.append(0)
        turbidez_centrifugado_1.append(0)
cineticos_centrifugacion = cineticos_centrifugacion.drop(columns=['Centrifugada (1 o 2)', 'Volumen centrifugado (L)', 'Turbidez'])
cineticos_centrifugacion['Volumen_centrifugado_1'] = vol_centrifugado_1
cineticos_centrifugacion['Volumen_centrifugado_2'] = vol_centrifugado_2
cineticos_centrifugacion['Turbidez_centrifugado_1'] = turbidez_centrifugado_1
cineticos_centrifugacion['Turbidez_centrifugado_2'] = turbidez_centrifugado_2
# Pasamos las columnas nuevas a float
cineticos_centrifugacion['Volumen_centrifugado_1'] = pd.to_numeric(cineticos_centrifugacion['Volumen_centrifugado_1'], errors='coerce')
cineticos_centrifugacion['Volumen_centrifugado_1'] = cineticos_centrifugacion['Volumen_centrifugado_1'].astype(float)
cineticos_centrifugacion['Volumen_centrifugado_1'] = cineticos_centrifugacion['Volumen_centrifugado_1'].fillna(method='ffill')
cineticos_centrifugacion['Volumen_centrifugado_2'] = pd.to_numeric(cineticos_centrifugacion['Volumen_centrifugado_2'], errors='coerce')
cineticos_centrifugacion['Volumen_centrifugado_2'] = cineticos_centrifugacion['Volumen_centrifugado_2'].astype(float)
cineticos_centrifugacion['Volumen_centrifugado_2'] = cineticos_centrifugacion['Volumen_centrifugado_2'].fillna(method='ffill')
cineticos_centrifugacion['Turbidez_centrifugado_1'] = pd.to_numeric(cineticos_centrifugacion['Turbidez_centrifugado_1'], errors='coerce')
cineticos_centrifugacion['Turbidez_centrifugado_1'] = cineticos_centrifugacion['Turbidez_centrifugado_1'].astype(float)
cineticos_centrifugacion['Turbidez_centrifugado_1'] = cineticos_centrifugacion['Turbidez_centrifugado_1'].fillna(method='ffill')
cineticos_centrifugacion['Turbidez_centrifugado_2'] = pd.to_numeric(cineticos_centrifugacion['Turbidez_centrifugado_2'], errors='coerce')
cineticos_centrifugacion['Turbidez_centrifugado_2'] = cineticos_centrifugacion['Turbidez_centrifugado_2'].astype(float)
cineticos_centrifugacion['Turbidez_centrifugado_2'] = cineticos_centrifugacion['Turbidez_centrifugado_2'].fillna(method='ffill')
# Ahora sacamos el minimum, mean y maximum de los valores de turbidez y volumen de cada lote
lotes = cineticos_centrifugacion['Lote'].unique()
vol_centrifugado_1_min = []
vol_centrifugado_1_mean = []
vol_centrifugado_1_max = []
vol_centrifugado_2_min = []
vol_centrifugado_2_mean = []
vol_centrifugado_2_max = []
turbidez_centrifugado_1_min = []
turbidez_centrifugado_1_mean = []
turbidez_centrifugado_1_max = []
turbidez_centrifugado_2_min = []
turbidez_centrifugado_2_mean = []
turbidez_centrifugado_2_max = []
for lote in lotes:
    cinetico = cineticos_centrifugacion[cineticos_centrifugacion['Lote'] == lote]
    vol_centrifugado_1_min.append(cinetico['Volumen_centrifugado_1'].min())
    vol_centrifugado_1_mean.append(cinetico['Volumen_centrifugado_1'].mean())
    vol_centrifugado_1_max.append(cinetico['Volumen_centrifugado_1'].max())
    vol_centrifugado_2_min.append(cinetico['Volumen_centrifugado_2'].min())
    vol_centrifugado_2_mean.append(cinetico['Volumen_centrifugado_2'].mean())
    vol_centrifugado_2_max.append(cinetico['Volumen_centrifugado_2'].max())
    turbidez_centrifugado_1_min.append(cinetico['Turbidez_centrifugado_1'].min())
    turbidez_centrifugado_1_mean.append(cinetico['Turbidez_centrifugado_1'].mean())
    turbidez_centrifugado_1_max.append(cinetico['Turbidez_centrifugado_1'].max())
    turbidez_centrifugado_2_min.append(cinetico['Turbidez_centrifugado_2'].min())
    turbidez_centrifugado_2_mean.append(cinetico['Turbidez_centrifugado_2'].mean())
    turbidez_centrifugado_2_max.append(cinetico['Turbidez_centrifugado_2'].max())
# Creamos un dataframe con los valores
cineticos_centrifugacion = pd.DataFrame({'Lote': lotes, 'Volumen_centrifugado_1_min': vol_centrifugado_1_min, 'Volumen_centrifugado_1_mean': vol_centrifugado_1_mean, 'Volumen_centrifugado_1_max': vol_centrifugado_1_max, 'Volumen_centrifugado_2_min': vol_centrifugado_2_min, 'Volumen_centrifugado_2_mean': vol_centrifugado_2_mean, 'Volumen_centrifugado_2_max': vol_centrifugado_2_max, 'Turbidez_centrifugado_1_min': turbidez_centrifugado_1_min, 'Turbidez_centrifugado_1_mean': turbidez_centrifugado_1_mean, 'Turbidez_centrifugado_1_max': turbidez_centrifugado_1_max, 'Turbidez_centrifugado_2_min': turbidez_centrifugado_2_min, 'Turbidez_centrifugado_2_mean': turbidez_centrifugado_2_mean, 'Turbidez_centrifugado_2_max': turbidez_centrifugado_2_max})
cineticos_centrifugacion = cineticos_centrifugacion.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de cineticos_centrifugacion es: ", cineticos_centrifugacion.shape)
# Nos guardamos el cineticos_centrifugacion
cineticos_centrifugacion.to_csv('./processed_data/cineticos_centrifugacion.csv', index=False)


# In[30]:


#Movimientos componentes.xlsx
movimientos = pd.read_excel('data/Movimientos componentes.xlsx')
print("El número de filas y columnas de movimientos es: ", movimientos.shape)
try:
    movimientos['Lote'] = movimientos['Lote'].apply(lambda x: int(x))
except:
    # Si no se puede pasar a int, es que hay un valor que no es un número
    # y por ende se tiene que eliminar
    movimientos = movimientos[movimientos['Lote'] != 'Lote']
print("El número de filas y columnas de movimientos es: ", movimientos.shape)
# OFs
of = pd.read_csv('./processed_data/of.csv')
# Nos quedamos con el lote y lo pasamos a int
of = of[['Lote']]
of['Lote'] = of['Lote'].apply(lambda x: int(x))
# Solo usamos los lotes que están en movimientos
movimientos = movimientos[movimientos['Lote'].isin(of['Lote'])]
print("El número de filas y columnas de movimientos es: ", movimientos.shape)
# Para cada lote nos guardamos la columna Qty, que es un indicador de calidad del producto
lotes = movimientos['Lote'].unique()
qty = []
for lote in lotes:
    mov = movimientos[movimientos['Lote'] == lote]
    qty.append(mov['Qty'].mean())
# Creamos un dataframe con los valores
movimientos = pd.DataFrame({'Lote': lotes, 'Qty': qty})
movimientos = movimientos.drop_duplicates(subset=['Lote'])
print("El número de filas y columnas de movimientos es: ", movimientos.shape)
# Nos guardamos el movimientos
movimientos.to_csv('./processed_data/movimientos.csv', index=False)


# In[31]:


# Ahora creamos una función que nos permita hacer lo mismo para todos los bioreactores con diferentes ID
def preprocesing_bioreactor(id_biorreactor):
    biorreactor = pd.read_excel('data/Biorreactores/Biorreactor ' + str(id_biorreactor) + '.xlsx', sheet_name='Datos')
    biorreactor = biorreactor.rename(columns={'DateTime': 'F_h_bio_in'})
    biorreactor['F_h_bio_in'] = biorreactor['F_h_bio_in'].apply(lambda x: x[:-4])
    biorreactor['F_h_bio_in'] = pd.to_datetime(biorreactor['F_h_bio_in'], format='%Y-%m-%d %H:%M:%S')
    id_biorreactor = str(id_biorreactor)
    biorreactor = biorreactor.rename(columns={id_biorreactor + '_FERM0101.Agitation_PV': 'Agitation_PV', id_biorreactor + '_FERM0101.Air_Sparge_PV': 'Air_Sparge_PV', id_biorreactor + '_FERM0101.Biocontainer_Pressure_PV': 'Pressure_PV', id_biorreactor + '_FERM0101.DO_1_PV': 'DO_1_PV', id_biorreactor + '_FERM0101.DO_2_PV': 'DO_2_PV', id_biorreactor + '_FERM0101.Gas_Overlay_PV': 'Gas_Overlay_PV', id_biorreactor + '_FERM0101.Load_Cell_Net_PV': 'Load_Cell_Net_PV', id_biorreactor + '_FERM0101.pH_1_PV': 'pH_1_PV', id_biorreactor + '_FERM0101.pH_2_PV': 'pH_2_PV'})
    biorreactor = biorreactor.rename(columns={id_biorreactor + '_FERM0101.PUMP_1_PV': 'PUMP_1_PV', id_biorreactor + '_FERM0101.PUMP_2_PV': 'PUMP_2_PV', id_biorreactor + '_FERM0101.PUMP_1_TOTAL': 'PUMP_1_TOTAL', id_biorreactor + '_FERM0101.PUMP_2_TOTAL': 'PUMP_2_TOTAL', id_biorreactor + '_FERM0101.Single_Use_DO_PV': 'Single_Use_DO_PV', id_biorreactor + '_FERM0101.Single_Use_pH_PV': 'Single_Use_pH_PV', id_biorreactor + '_FERM0101.Temperatura_PV': 'Temperatura_PV'})
    biorreactor['Agitation_PV'] = pd.to_numeric(biorreactor['Agitation_PV'], errors='coerce')
    biorreactor['Agitation_PV'] = biorreactor['Agitation_PV'].astype(float)
    biorreactor['Agitation_PV'] = biorreactor['Agitation_PV'].fillna(method='ffill')
    biorreactor['Air_Sparge_PV'] = pd.to_numeric(biorreactor['Air_Sparge_PV'], errors='coerce')
    biorreactor['Air_Sparge_PV'] = biorreactor['Air_Sparge_PV'].astype(float)
    biorreactor['Air_Sparge_PV'] = biorreactor['Air_Sparge_PV'].fillna(method='ffill')
    biorreactor['Pressure_PV'] = pd.to_numeric(biorreactor['Pressure_PV'], errors='coerce')
    biorreactor['Pressure_PV'] = biorreactor['Pressure_PV'].astype(float)
    biorreactor['Pressure_PV'] = biorreactor['Pressure_PV'].fillna(method='ffill')
    biorreactor['DO_1_PV'] = pd.to_numeric(biorreactor['DO_1_PV'], errors='coerce')
    biorreactor['DO_1_PV'] = biorreactor['DO_1_PV'].astype(float)
    biorreactor['DO_1_PV'] = biorreactor['DO_1_PV'].fillna(method='ffill')
    biorreactor['DO_2_PV'] = pd.to_numeric(biorreactor['DO_2_PV'], errors='coerce')
    biorreactor['DO_2_PV'] = biorreactor['DO_2_PV'].astype(float)
    biorreactor['DO_2_PV'] = biorreactor['DO_2_PV'].fillna(method='ffill')
    biorreactor['Gas_Overlay_PV'] = pd.to_numeric(biorreactor['Gas_Overlay_PV'], errors='coerce')
    biorreactor['Gas_Overlay_PV'] = biorreactor['Gas_Overlay_PV'].astype(float)
    biorreactor['Gas_Overlay_PV'] = biorreactor['Gas_Overlay_PV'].fillna(method='ffill')
    biorreactor['Load_Cell_Net_PV'] = pd.to_numeric(biorreactor['Load_Cell_Net_PV'], errors='coerce')
    biorreactor['Load_Cell_Net_PV'] = biorreactor['Load_Cell_Net_PV'].astype(float)
    biorreactor['Load_Cell_Net_PV'] = biorreactor['Load_Cell_Net_PV'].fillna(method='ffill')
    biorreactor['pH_1_PV'] = pd.to_numeric(biorreactor['pH_1_PV'], errors='coerce')
    biorreactor['pH_1_PV'] = biorreactor['pH_1_PV'].astype(float)
    biorreactor['pH_1_PV'] = biorreactor['pH_1_PV'].fillna(method='ffill')
    biorreactor['pH_2_PV'] = pd.to_numeric(biorreactor['pH_2_PV'], errors='coerce')
    biorreactor['pH_2_PV'] = biorreactor['pH_2_PV'].astype(float)
    biorreactor['pH_2_PV'] = biorreactor['pH_2_PV'].fillna(method='ffill')
    biorreactor['PUMP_1_PV'] = pd.to_numeric(biorreactor['PUMP_1_PV'], errors='coerce')
    biorreactor['PUMP_1_PV'] = biorreactor['PUMP_1_PV'].astype(float)
    biorreactor['PUMP_1_PV'] = biorreactor['PUMP_1_PV'].fillna(method='ffill')
    biorreactor['PUMP_2_PV'] = pd.to_numeric(biorreactor['PUMP_2_PV'], errors='coerce')
    biorreactor['PUMP_2_PV'] = biorreactor['PUMP_2_PV'].astype(float)
    biorreactor['PUMP_2_PV'] = biorreactor['PUMP_2_PV'].fillna(method='ffill')
    biorreactor['PUMP_1_TOTAL'] = pd.to_numeric(biorreactor['PUMP_1_TOTAL'], errors='coerce')
    biorreactor['PUMP_1_TOTAL'] = biorreactor['PUMP_1_TOTAL'].astype(float)
    biorreactor['PUMP_1_TOTAL'] = biorreactor['PUMP_1_TOTAL'].fillna(method='ffill')
    biorreactor['PUMP_2_TOTAL'] = pd.to_numeric(biorreactor['PUMP_2_TOTAL'], errors='coerce')
    biorreactor['PUMP_2_TOTAL'] = biorreactor['PUMP_2_TOTAL'].astype(float)
    biorreactor['PUMP_2_TOTAL'] = biorreactor['PUMP_2_TOTAL'].fillna(method='ffill')
    biorreactor['Single_Use_DO_PV'] = pd.to_numeric(biorreactor['Single_Use_DO_PV'], errors='coerce')
    biorreactor['Single_Use_DO_PV'] = biorreactor['Single_Use_DO_PV'].astype(float)
    biorreactor['Single_Use_DO_PV'] = biorreactor['Single_Use_DO_PV'].fillna(method='ffill')
    biorreactor['Single_Use_pH_PV'] = pd.to_numeric(biorreactor['Single_Use_pH_PV'], errors='coerce')
    biorreactor['Single_Use_pH_PV'] = biorreactor['Single_Use_pH_PV'].astype(float)
    biorreactor['Single_Use_pH_PV'] = biorreactor['Single_Use_pH_PV'].fillna(method='ffill')
    biorreactor['Temperatura_PV'] = pd.to_numeric(biorreactor['Temperatura_PV'], errors='coerce')
    biorreactor['Temperatura_PV'] = biorreactor['Temperatura_PV'].astype(float)
    biorreactor['Temperatura_PV'] = biorreactor['Temperatura_PV'].fillna(method='ffill')
    biorreactor['ID_biorreactor'] = id_biorreactor
    biorreactor = biorreactor.sort_values(by=['ID_biorreactor'])
    biorreactor['ID_biorreactor'] = biorreactor['ID_biorreactor'].astype(int)
    # Ordenamos por fecha
    biorreactor = biorreactor.sort_values(by=['F_h_bio_in'])
    biorreactor.to_csv('./processed_data/biorreactores/biorreactor_' + str(id_biorreactor) + '.csv', index=False)
# Con los IDs de los bioreactores únicos, vamos a leerlos de data/Biorreactor xxxxx.xlsx
bioreactores = [13171, 13172, 14618]
for bioreactor in bioreactores:
    print('Bioreactor: ', bioreactor)
    preprocesing_bioreactor(bioreactor)
# Combinamos todos los bioreactores en un solo dataframe que llamaremos bioreactores_pequeños
bioreactores_pequeños = pd.DataFrame()
for bioreactor in bioreactores:
    tmp = pd.read_csv('./processed_data/biorreactores/biorreactor_' + str(bioreactor) + '.csv')
    bioreactores_pequeños = pd.concat([bioreactores_pequeños, tmp])
bioreactores_pequeños.to_csv('./processed_data/biorreactores/biorreactores_pequeños.csv', index=False)
# Con los IDs de los bioreactores únicos, vamos a leerlos de data/Biorreactor xxxxx.xlsx
bioreactores = [13169, 13170, 14614, 14615, 14616, 14617]
for bioreactor in bioreactores:
    print('Bioreactor: ', bioreactor)
    preprocesing_bioreactor(bioreactor)
# Combinamos todos los bioreactores en un solo dataframe que llamaremos bioreactores_pequeños
bioreactores_grandes = pd.DataFrame()
for bioreactor in bioreactores:
    tmp = pd.read_csv('./processed_data/biorreactores/biorreactor_' + str(bioreactor) + '.csv')
    bioreactores_grandes = pd.concat([bioreactores_grandes, tmp])
bioreactores_grandes.to_csv('./processed_data/biorreactores/bioreactores_grandes.csv', index=False)


# In[32]:


def preprocesing_centrifuga(id_centrifuga):
    centrifuga = pd.read_excel('data/Centrifugadoras/Centri╠üfuga ' + str(id_centrifuga) + '.xlsx', sheet_name='Datos')
    # Cambiamos el nombre de DateTime a Fecha
    centrifuga = centrifuga.rename(columns={'DateTime': 'F_h_cen_cul'})
    centrifuga['F_h_cen_cul'] = centrifuga['F_h_cen_cul'].apply(lambda x: x[:-4])
    centrifuga['F_h_cen_cul'] = pd.to_datetime(centrifuga['F_h_cen_cul'], format='%Y-%m-%d %H:%M:%S')
    # Pasamos todas las columnas a float menos la fecha
    columns = centrifuga.columns
    for column in columns:
        if column != 'F_h_cen_cul':
            centrifuga[column] = pd.to_numeric(centrifuga[column], errors='coerce')
            centrifuga[column] = centrifuga[column].astype(float)
            # Si hay algún valor NaN, lo cambiamos usando forward fill
            centrifuga[column] = centrifuga[column].fillna(method='ffill')
    # Finalmente añadimos una columna con el ID de la centrífuga
    centrifuga['ID_centrifuga'] = id_centrifuga
    # Ordenamos por Fecha
    centrifuga = centrifuga.sort_values(by=['F_h_cen_cul'])
    centrifuga
    # Guardamos el dataframe en un csv
    centrifuga.to_csv('./processed_data/centrifugadoras/centrifuga_' + str(id_centrifuga) + '.csv', index=False)

centrifugadoras = [12912,14246,17825]
for centrifuga in centrifugadoras:
    print('Centrifuga: ', centrifuga)
    preprocesing_centrifuga(centrifuga)

# Combinamos todos los bioreactores en un solo dataframe que llamaremos bioreactores_pequeños
centrifugadoras_cultivo = pd.DataFrame()
for centrifuga in centrifugadoras:
    tmp = pd.read_csv('./processed_data/centrifugadoras/centrifuga_' + str(centrifuga) + '.csv')
    print("Shape: ", tmp.shape)
    centrifugadoras_cultivo = pd.concat([centrifugadoras_cultivo, tmp])
    print("Shape after: ", centrifugadoras_cultivo.shape)
centrifugadoras_cultivo.to_csv('./processed_data/centrifugadoras/centrifugadoras.csv', index=False)


# In[34]:


temperatura_humedad = pd.read_excel('data/Temperaturas y humedades.xlsx', sheet_name='Datos')
temperatura_humedad = temperatura_humedad.rename(columns={'DateTime': 'F_h_cen_tem_hum'})
temperatura_humedad['F_h_cen_tem_hum'] = temperatura_humedad['F_h_cen_tem_hum'].apply(lambda x: x[:-4])
temperatura_humedad['F_h_cen_tem_hum'] = pd.to_datetime(temperatura_humedad['F_h_cen_tem_hum'], format='%Y-%m-%d %H:%M:%S')
columns = temperatura_humedad.columns
for column in columns:
    if column != 'F_h_cen_tem_hum':
        temperatura_humedad[column] = pd.to_numeric(temperatura_humedad[column], errors='coerce')
        temperatura_humedad[column] = temperatura_humedad[column].astype(float)
        # Si hay algún valor NaN, lo cambiamos usando forward fill
        temperatura_humedad[column] = temperatura_humedad[column].fillna(method='ffill')
# Cambiamos el nombre de las columnas
temperatura_humedad = temperatura_humedad.rename(columns={'06299_TI1302.PV': 'temp_bio','06299_MI1302.PV': 'hum_bio','06299_TI1402.PV': 'temp_cen','06299_MI1402.PV': 'hum_cen','07633_TI0601.PV': 'temp_almacen','07633_HI0101.PV': 'hum_almacen', '07781_TI1501.PV': 'temp_produccion','07781_MI1501.PV': 'hum_produccion'})
temperatura_humedad.to_csv('./processed_data/temperatura_humedad.csv', index=False)
temperatura_humedad


# ## Matching Time 🕒

# ### Aqui modificamos las filas de preinóculo e inóculo para que tengan en cuenta lo del lote parental

# In[35]:


# Empezamos leyendo preinoculos y lo macheamos con inoculo 
preinoculos = pd.read_csv('./processed_data/preinoculo.csv') # contiene 165 filas
inoculos = pd.read_csv('./processed_data/inoculo.csv') # contiene 168 filas
cultivos = pd.read_csv('./processed_data/cultivo.csv') # contiene 168 filas
cultivos['Lote_parental'] = pd.to_numeric(cultivos['Lote_parental'], errors='coerce')
cultivos['Lote_parental'] = cultivos['Lote_parental'].astype('Int64')
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
lotes_cultivo = cultivos['Lote'].unique() # Obtenemos los lotes de cultivo
for lote in lotes_cultivo:
    tmp = cultivos[cultivos['Lote'] == lote]
    if tmp['orden_encadenado'].values[0] == 2:
        # Creamos una nueva fila en preinoculo
        tmp2 = preinoculos[preinoculos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp2['Lote'] = lote
        preinoculos = pd.concat([preinoculos, tmp2])
        # Ahora en inoculo
        tmp2 = inoculos[inoculos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp2['Lote'] = lote
        inoculos = pd.concat([inoculos, tmp2])
    # EN caso de que sea una orden encadenado 3, miramos el lote parental del lote parental
    if tmp['orden_encadenado'].values[0] == 3:
        tmp2 = cultivos[cultivos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp3 = preinoculos[preinoculos['Lote'] == tmp2['Lote_parental'].values[0]]
        tmp3['Lote'] = lote
        preinoculos = pd.concat([preinoculos, tmp3])
        # Ahora en inoculo
        tmp3 = inoculos[inoculos['Lote'] == tmp2['Lote_parental'].values[0]]
        tmp3['Lote'] = lote
        inoculos = pd.concat([inoculos, tmp3])
preinoculos = preinoculos.drop_duplicates(subset=['Lote'])
inoculos = inoculos.drop_duplicates(subset=['Lote'])
cultivos = cultivos.drop_duplicates(subset=['Lote'])
print("\n")
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
preinoculos.to_csv('./processed_data/preinoculo.csv', index=False)
inoculos.to_csv('./processed_data/inoculo.csv', index=False)


# In[36]:


# Empezamos leyendo preinoculos y lo macheamos con inoculo 
preinoculos = pd.read_csv('./processed_data/preinoculo.csv') # contiene 165 filas
inoculos = pd.read_csv('./processed_data/inoculo.csv') # contiene 168 filas
cultivos = pd.read_csv('./processed_data/cultivo_test.csv') # contiene 168 filas
cultivos['Lote_parental'] = pd.to_numeric(cultivos['Lote_parental'], errors='coerce')
cultivos['Lote_parental'] = cultivos['Lote_parental'].astype('Int64')
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
lotes_cultivo = cultivos['Lote'].unique() # Obtenemos los lotes de cultivo
for lote in lotes_cultivo:
    tmp = cultivos[cultivos['Lote'] == lote]
    if tmp['orden_encadenado'].values[0] == 2:
        # Creamos una nueva fila en preinoculo
        tmp2 = preinoculos[preinoculos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp2['Lote'] = lote
        preinoculos = pd.concat([preinoculos, tmp2])
        # Ahora en inoculo
        tmp2 = inoculos[inoculos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp2['Lote'] = lote
        inoculos = pd.concat([inoculos, tmp2])
    # EN caso de que sea una orden encadenado 3, miramos el lote parental del lote parental
    if tmp['orden_encadenado'].values[0] == 3:
        tmp2 = cultivos[cultivos['Lote'] == tmp['Lote_parental'].values[0]]
        tmp3 = preinoculos[preinoculos['Lote'] == tmp2['Lote_parental'].values[0]]
        tmp3['Lote'] = lote
        preinoculos = pd.concat([preinoculos, tmp3])
        # Ahora en inoculo
        tmp3 = inoculos[inoculos['Lote'] == tmp2['Lote_parental'].values[0]]
        tmp3['Lote'] = lote
        inoculos = pd.concat([inoculos, tmp3])
preinoculos = preinoculos.drop_duplicates(subset=['Lote'])
inoculos = inoculos.drop_duplicates(subset=['Lote'])
cultivos = cultivos.drop_duplicates(subset=['Lote'])
print("\n")
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
preinoculos.to_csv('./processed_data/preinoculo.csv', index=False)
inoculos.to_csv('./processed_data/inoculo.csv', index=False)


# ### Aqui añadimos al cultivo los centrifuagados y los biorreactores, además de los cinéticos

# In[37]:


centrifugadoras = pd.read_csv('./processed_data/centrifugadoras/centrifugadoras.csv')

# Ahora hacemos lo mismo para el cultivo y las centrifugadoras
cultivos = pd.read_csv('./processed_data/cultivo.csv')
# Creamos nuevas columnas en cultivo que será todas las que hay en centrifugadoras, menos F_h_cen_cul y ID_centrifuga
columnas_nuevas = centrifugadoras.columns.drop(['F_h_cen_cul', 'ID_centrifuga'])
for columna in columnas_nuevas:
    cultivos[columna + '_min'] = np.nan
    cultivos[columna + '_max'] = np.nan
    cultivos[columna + '_mean'] = np.nan
for i in range(0, len(cultivos)):
    fecha_inicio = cultivos.loc[i, 'F_h_init_cul']
    fecha_fin = cultivos.loc[i, 'F_h_end_cul']
    if pd.isnull(fecha_inicio) or pd.isnull(fecha_fin):
        continue
    else:
        # Ahora buscamos en el csv de centrifugadoras el cultivo y creamos un nuevo dataframe con tantas columnas como el csv de centrifugadoras
        tmp = centrifugadoras[(centrifugadoras['F_h_cen_cul'] >= fecha_inicio) & (centrifugadoras['F_h_cen_cul'] <= fecha_fin)]
        # Rellenamos las columnas de cultivo
        if len(tmp) > 0:
            for columna in columnas_nuevas:
                cultivos.loc[i, columna + '_min'] = tmp[columna].min()
                cultivos.loc[i, columna + '_max'] = tmp[columna].max()
                cultivos.loc[i, columna + '_mean'] = tmp[columna].mean()

bioreactores_grandes = pd.read_csv('./processed_data/biorreactores/bioreactores_grandes.csv')

columnas_nuevas = bioreactores_grandes.columns.drop(['F_h_bio_in', 'ID_biorreactor'])
for columna in columnas_nuevas:
    cultivos[columna + '_min'] = np.nan
    cultivos[columna + '_max'] = np.nan
    cultivos[columna + '_mean'] = np.nan
for i in range(0, len(cultivos)):
    fecha_inicio = cultivos.loc[i, 'F_h_init_cul']
    fecha_fin = cultivos.loc[i, 'F_h_end_cul']
    if pd.isnull(fecha_inicio) or pd.isnull(fecha_fin):
        continue
    else:
        # Ahora buscamos en el csv de centrifugadoras el cultivo y creamos un nuevo dataframe con tantas columnas como el csv de centrifugadoras
        tmp = bioreactores_grandes[(bioreactores_grandes['F_h_bio_in'] >= fecha_inicio) & (bioreactores_grandes['F_h_bio_in'] <= fecha_fin)]
        # Rellenamos las columnas de cultivo
        if len(tmp) > 0:
            for columna in columnas_nuevas:
                cultivos.loc[i, columna + '_min'] = tmp[columna].min()
                cultivos.loc[i, columna + '_max'] = tmp[columna].max()
                cultivos.loc[i, columna + '_mean'] = tmp[columna].mean()
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cineticos_cultivos = pd.read_csv('./processed_data/cineticos_cultivos.csv')
lotes = cultivos['Lote'].unique()
cineticos_cultivos = cineticos_cultivos[cineticos_cultivos['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, cineticos_cultivos, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cineticos_centrifugacion = pd.read_csv('./processed_data/cineticos_centrifugacion.csv')
lotes = cultivos['Lote'].unique()
cineticos_centrifugacion = cineticos_centrifugacion[cineticos_centrifugacion['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, cineticos_centrifugacion, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
movimientos = pd.read_csv('./processed_data/movimientos.csv')
lotes = cultivos['Lote'].unique()
movimientos = movimientos[movimientos['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, movimientos, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cultivos.to_csv('./processed_data/train/cultivos.csv', index=False)


# In[38]:


centrifugadoras = pd.read_csv('./processed_data/centrifugadoras/centrifugadoras.csv')

# Ahora hacemos lo mismo para el cultivo y las centrifugadoras
cultivos = pd.read_csv('./processed_data/cultivo_test.csv')
# Creamos nuevas columnas en cultivo que será todas las que hay en centrifugadoras, menos F_h_cen_cul y ID_centrifuga
columnas_nuevas = centrifugadoras.columns.drop(['F_h_cen_cul', 'ID_centrifuga'])
for columna in columnas_nuevas:
    cultivos[columna + '_min'] = np.nan
    cultivos[columna + '_max'] = np.nan
    cultivos[columna + '_mean'] = np.nan
for i in range(0, len(cultivos)):
    fecha_inicio = cultivos.loc[i, 'F_h_init_cul']
    fecha_fin = cultivos.loc[i, 'F_h_end_cul']
    if pd.isnull(fecha_inicio) or pd.isnull(fecha_fin):
        continue
    else:
        # Ahora buscamos en el csv de centrifugadoras el cultivo y creamos un nuevo dataframe con tantas columnas como el csv de centrifugadoras
        tmp = centrifugadoras[(centrifugadoras['F_h_cen_cul'] >= fecha_inicio) & (centrifugadoras['F_h_cen_cul'] <= fecha_fin)]
        # Rellenamos las columnas de cultivo
        if len(tmp) > 0:
            for columna in columnas_nuevas:
                cultivos.loc[i, columna + '_min'] = tmp[columna].min()
                cultivos.loc[i, columna + '_max'] = tmp[columna].max()
                cultivos.loc[i, columna + '_mean'] = tmp[columna].mean()

bioreactores_grandes = pd.read_csv('./processed_data/biorreactores/bioreactores_grandes.csv')

columnas_nuevas = bioreactores_grandes.columns.drop(['F_h_bio_in', 'ID_biorreactor'])
for columna in columnas_nuevas:
    cultivos[columna + '_min'] = np.nan
    cultivos[columna + '_max'] = np.nan
    cultivos[columna + '_mean'] = np.nan
for i in range(0, len(cultivos)):
    fecha_inicio = cultivos.loc[i, 'F_h_init_cul']
    fecha_fin = cultivos.loc[i, 'F_h_end_cul']
    if pd.isnull(fecha_inicio) or pd.isnull(fecha_fin):
        continue
    else:
        # Ahora buscamos en el csv de centrifugadoras el cultivo y creamos un nuevo dataframe con tantas columnas como el csv de centrifugadoras
        tmp = bioreactores_grandes[(bioreactores_grandes['F_h_bio_in'] >= fecha_inicio) & (bioreactores_grandes['F_h_bio_in'] <= fecha_fin)]
        # Rellenamos las columnas de cultivo
        if len(tmp) > 0:
            for columna in columnas_nuevas:
                cultivos.loc[i, columna + '_min'] = tmp[columna].min()
                cultivos.loc[i, columna + '_max'] = tmp[columna].max()
                cultivos.loc[i, columna + '_mean'] = tmp[columna].mean()
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cineticos_cultivos = pd.read_csv('./processed_data/cineticos_cultivos.csv')
lotes = cultivos['Lote'].unique()
cineticos_cultivos = cineticos_cultivos[cineticos_cultivos['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, cineticos_cultivos, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cineticos_centrifugacion = pd.read_csv('./processed_data/cineticos_centrifugacion.csv')
lotes = cultivos['Lote'].unique()
cineticos_centrifugacion = cineticos_centrifugacion[cineticos_centrifugacion['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, cineticos_centrifugacion, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
movimientos = pd.read_csv('./processed_data/movimientos.csv')
lotes = cultivos['Lote'].unique()
movimientos = movimientos[movimientos['Lote'].isin(lotes)]
# Hacemos el merge
cultivos = pd.merge(cultivos, movimientos, on='Lote', how='left')
print("El número de filas y columnas de cultivos es: ", cultivos.shape)
cultivos.to_csv('./processed_data/test/cultivos.csv', index=False)


# ### Aqui vamos a hacer los biorreactores de inoculo y su cinético

# In[39]:


inoculos = pd.read_csv('./processed_data/inoculo.csv')
bioreactores_pequeños = pd.read_csv('./processed_data/biorreactores/biorreactores_pequeños.csv')

columnas_nuevas = bioreactores_pequeños.columns.drop(['F_h_bio_in', 'ID_biorreactor'])
for columna in columnas_nuevas:
    inoculos[columna + '_min'] = np.nan
    inoculos[columna + '_max'] = np.nan
    inoculos[columna + '_mean'] = np.nan
for i in range(0, len(inoculos)): #F_h_init_in
    fecha_inicio = inoculos.loc[i, 'F_h_init_in']
    fecha_fin = inoculos.loc[i, 'F_h_end_in']
    if pd.isnull(fecha_inicio) or pd.isnull(fecha_fin):
        continue
    else:
        # Ahora buscamos en el csv de centrifugadoras el cultivo y creamos un nuevo dataframe con tantas columnas como el csv de centrifugadoras
        tmp = bioreactores_pequeños[(bioreactores_pequeños['F_h_bio_in'] >= fecha_inicio) & (bioreactores_pequeños['F_h_bio_in'] <= fecha_fin)]
        # Rellenamos las columnas de cultivo
        if len(tmp) > 0:
            for columna in columnas_nuevas:
                inoculos.loc[i, columna + '_min'] = tmp[columna].min()
                inoculos.loc[i, columna + '_max'] = tmp[columna].max()
                inoculos.loc[i, columna + '_mean'] = tmp[columna].mean()
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
cineticos_inoculos = pd.read_csv('./processed_data/cineticos_inoculos.csv')
lotes = inoculos['Lote'].unique()
cineticos_inoculos = cineticos_inoculos[cineticos_inoculos['Lote'].isin(lotes)]
# Hacemos el merge
inoculos = pd.merge(inoculos, cineticos_inoculos, on='Lote', how='left')
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
inoculos.to_csv('./processed_data/inoculo.csv', index=False)


# In[40]:


preinoculos = pd.read_csv('./processed_data/preinoculo.csv')
inoculos = pd.read_csv('./processed_data/inoculo.csv')
cultivos = pd.read_csv('./processed_data/train/cultivos.csv')
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)

preinoculos['Lote'] = preinoculos['Lote'].astype('Int64')
inoculos['Lote'] = inoculos['Lote'].astype('Int64')
cultivos['Lote'] = cultivos['Lote'].astype('Int64')
preinoculos = preinoculos.sort_values(by=['Lote'])
inoculos = inoculos.sort_values(by=['Lote'])
cultivos = cultivos.sort_values(by=['Lote'])
lotes_cultivo = cultivos['Lote'].unique()
preinoculos_cultivo = preinoculos[preinoculos['Lote'].isin(lotes_cultivo)]
inoculos_cultivo = inoculos[inoculos['Lote'].isin(lotes_cultivo)]

print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
print("El número de filas y columnas de inoculos_cultivo es: ", inoculos_cultivo.shape)
# Hacemos un merge de preinoculos_cultivo con inoculos_cultivo
preinoculos_cultivo = pd.merge(preinoculos_cultivo, inoculos_cultivo, on='Lote', how='left')
print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
# Hacemos un merge de preinoculos_cultivo con cultivos
preinoculos_cultivo = pd.merge(preinoculos_cultivo, cultivos, on='Lote', how='left')
print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
preinoculos_cultivo.to_csv('./processed_data/train/train_data.csv', index=False)


# In[41]:


preinoculos = pd.read_csv('./processed_data/preinoculo.csv')
inoculos = pd.read_csv('./processed_data/inoculo.csv')
cultivos = pd.read_csv('./processed_data/test/cultivos.csv')
print("El número de filas y columnas de preinoculos es: ", preinoculos.shape)
print("El número de filas y columnas de inoculos es: ", inoculos.shape)
print("El número de filas y columnas de cultivos es: ", cultivos.shape)

preinoculos['Lote'] = preinoculos['Lote'].astype('Int64')
inoculos['Lote'] = inoculos['Lote'].astype('Int64')
cultivos['Lote'] = cultivos['Lote'].astype('Int64')
preinoculos = preinoculos.sort_values(by=['Lote'])
inoculos = inoculos.sort_values(by=['Lote'])
cultivos = cultivos.sort_values(by=['Lote'])
lotes_cultivo = cultivos['Lote'].unique()
preinoculos_cultivo = preinoculos[preinoculos['Lote'].isin(lotes_cultivo)]
inoculos_cultivo = inoculos[inoculos['Lote'].isin(lotes_cultivo)]

print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
print("El número de filas y columnas de inoculos_cultivo es: ", inoculos_cultivo.shape)
# Hacemos un merge de preinoculos_cultivo con inoculos_cultivo
preinoculos_cultivo = pd.merge(preinoculos_cultivo, inoculos_cultivo, on='Lote', how='left')
print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
# Hacemos un merge de preinoculos_cultivo con cultivos
preinoculos_cultivo = pd.merge(preinoculos_cultivo, cultivos, on='Lote', how='left')
print("El número de filas y columnas de preinoculos_cultivo es: ", preinoculos_cultivo.shape)
preinoculos_cultivo.to_csv('./processed_data/test/test_data.csv', index=False)


# ## Investigación del train 🕵🏾
# 
# Generamos un informe de los datos de Entrenamiento para ver que tipo de datos tenemos y si hay valores nulos y que correlacion tienen los datos con la variable objetivo. Además de eliminar aquellas columnas que no aporten información relevante (como puede ser la fecha). También aprovechamos y pasamos todos los datos a float64 para que no haya problemas a la hora de entrenar el modelo.

# In[42]:


# Load the data
train_data = pd.read_csv('./processed_data/train/train_data.csv')
# We store in a txt the columns and the type of the columns
with open('./processed_data/train/columns.txt', 'w') as f:
    for col in train_data.columns:
        f.write(f'{col} - {train_data[col].dtype}\n')
# Drop all the object columns except those that starts with 'Durac'
import re
pattern = re.compile('Durac')
columns_to_drop = [col for col in train_data.columns if train_data[col].dtype == 'object' and not pattern.match(col)]
train_data.drop(columns=columns_to_drop, inplace=True)
# Now we parse the object columns to seconds, because they are in datetime format
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = pd.to_timedelta(train_data[col]).dt.total_seconds()
# We store the columns in a txt
with open('./processed_data/train/columns_clean.txt', 'a') as f:
    for col in train_data.columns:
        f.write(f'{col} - {train_data[col].dtype}\n')

# For each column we are going to store in a txt the mean, std, min, max and the number of NaN values, also the correlation with product 1
corr = train_data.corr()['Producto 1'].sort_values(ascending=False)
# Sort the columns by correlation
corr = corr[corr.index]
# We store the correlation in a txt
with open('./processed_data/train/stats.txt', 'w') as f:
    for col in corr.index:
        f.write(f'{col}\n')
        f.write(f'Mean: {train_data[col].mean()}\n')
        f.write(f'Std: {train_data[col].std()}\n')
        f.write(f'Min: {train_data[col].min()}\n')
        f.write(f'Max: {train_data[col].max()}\n')
        f.write(f'NaN values: {train_data[col].isna().sum()}\n')
        f.write(f'Correlation with Producto 1: {corr[col]}\n\n')


# Load the data
test_data = pd.read_csv('./processed_data/test/test_data.csv')
# We store in a txt the columns and the type of the columns
with open('./processed_data/test/columns.txt', 'w') as f:
    for col in test_data.columns:
        f.write(f'{col} - {test_data[col].dtype}\n')
# Drop all the object columns except those that starts with 'Durac'
import re
pattern = re.compile('Durac')
columns_to_drop = [col for col in test_data.columns if test_data[col].dtype == 'object' and not pattern.match(col)]
test_data.drop(columns=columns_to_drop, inplace=True)
# Now we parse the object columns to seconds, because they are in datetime format
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        test_data[col] = pd.to_timedelta(test_data[col]).dt.total_seconds()
# We store the columns in a txt
with open('./processed_data/test/columns_clean.txt', 'a') as f:
    for col in test_data.columns:
        f.write(f'{col} - {test_data[col].dtype}\n')

# For each column we are going to store in a txt the mean, std, min, max and the number of NaN values, also the correlation with product 1
#corr = train_data.corr()['Producto 1'].sort_values(ascending=False)
# Sort the columns by correlation
corr = corr[corr.index]
# We store the correlation in a txt
with open('./processed_data/test/stats.txt', 'w') as f:
    for col in corr.index:
        f.write(f'{col}\n')
        f.write(f'Mean: {test_data[col].mean()}\n')
        f.write(f'Std: {test_data[col].std()}\n')
        f.write(f'Min: {test_data[col].min()}\n')
        f.write(f'Max: {test_data[col].max()}\n')
        f.write(f'NaN values: {test_data[col].isna().sum()}\n')
        f.write(f'Correlation with Producto 1: {corr[col]}\n\n')


# In[43]:


# Store the data
train_data.to_csv('./processed_data/train/train_data_clean.csv', index=False)
test_data.to_csv('./processed_data/test/test_data_clean.csv', index=False)

