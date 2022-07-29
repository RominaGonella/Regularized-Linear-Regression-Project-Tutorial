# STEP 1: cargar los datos

# no olvidar ejecturar en consola: pip install -r requirements.txt

# librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# cargo datos
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')
df_raw.rename(columns = {'19-Oct':'10-19'}, inplace = True)

# guardo datos iniciales
df_raw.to_csv('data/raw/datos_iniciales.csv', index = False)

# STEP 2: análisis exploratorio

# copio datos a otro data frame con el cual trabajar
df = df_raw.copy()

# convierto a la clase correspondiente las categóricas
df.fips = pd.Categorical(df.fips)
df.COUNTY_NAME = pd.Categorical(df.COUNTY_NAME)
df.STATE_NAME = pd.Categorical(df.STATE_NAME)
df.STATE_FIPS = pd.Categorical(df.STATE_FIPS)
df.CNTY_FIPS = pd.Categorical(df.CNTY_FIPS)
df.Urban_rural_code = pd.Categorical(df.Urban_rural_code)

df = df.drop(columns = ['fips', 'COUNTY_NAME', 'STATE_NAME', 'STATE_FIPS', 'CNTY_FIPS'], axis = 1)

# elijo variable target (prevalencia de la obesidad) y separo en X e y
y = df['Obesity_prevalence']

elim = ['Obesity_prevalence', 'CI90LBINC_2018', 'CI90UBINC_2018', 'anycondition_prevalence',
'anycondition_Lower 95% CI', 'anycondition_Upper 95% CI', 'anycondition_number', 'Obesity_Lower 95% CI',
'Obesity_Upper 95% CI', 'Obesity_number', 'Heart disease_prevalence', 'Heart disease_Lower 95% CI',
'Heart disease_Upper 95% CI', 'Heart disease_number', 'COPD_prevalence', 'COPD_Lower 95% CI', 'COPD_Upper 95% CI',
'COPD_number', 'diabetes_prevalence', 'diabetes_Lower 95% CI', 'diabetes_Upper 95% CI', 'diabetes_number',
'CKD_prevalence', 'CKD_Lower 95% CI', 'CKD_Upper 95% CI', 'CKD_number']
X = df.drop(columns = elim, axis = 1)

# junto para guardar como base procesada
df_interim = pd.concat([X, y], axis = 1)
df_interim.to_csv('data/interim/datos_procesados.csv', index = False)

# separo en muestras de train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2607, test_size = 0.15)

# creo data frame de entrenamiento
df_train = pd.concat([X_train, y_train], axis = 1)

# STEP 3: aplicar LASSO

# se convierten categóricas a dummy
X_train = pd.get_dummies(X_train, drop_first = True)
X_test = pd.get_dummies(X_test, drop_first = True)

# estimo LASSO con ese valor de alpha
alfa_optimo = 0.0003409285069746815
mods_opt = Lasso(alpha = alfa_optimo, normalize = True)
mods_opt.fit(X_train, y_train)

# STEP 4: se guardan bases y modelo final

# se guardan datos de entrenamiento procesados
df_train.to_csv('data/processed/datos_entrenamiento.csv', index = False)

# se guarda el modelo
filename = 'models/reg_model.sav'
pickle.dump(mods_opt, open(filename,'wb'))