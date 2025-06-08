#!/usr/bin/env python
# coding: utf-8

# # DATA LOADER

# In[1]:


import pandas as pd 
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import StandardScaler




import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ruta correcta a los datos
train_data_path = "/mnt/data/train_data_clean.csv"

# Cargar datos
train_data = pd.read_csv(train_data_path)

# Separar X (features) y y (target)
X = train_data.drop(columns=["Producto 1"])  # Eliminar la columna objetivo
y = train_data["Producto 1"]

# Escalar los datos
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Guardar los datos preprocesados
X_scaled.to_csv("X_scaled.csv", index=False)


# # MODELADO PREDICTIVO

# In[2]:


import pandas as pd
import numpy as np
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
    GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# 📌 Configuración global
RANDOM_SEED = 42

# 📂 Cargar datos
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")

# 🏷️ Definir features y target
X = train_data.drop(columns=["Producto 1"])
y = train_data["Producto 1"]

# 🛠️ Imputación de valores faltantes y escalado
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 🔀 División en train y validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

# 📌 Definición de modelos y parámetros
models = {
    'LinearRegression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
    'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
    'ElasticNet': (ElasticNet(), {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
    'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 10], 'weights': ['uniform', 'distance']}),
    'SVR': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]}),
    'XGBoost': (XGBRegressor(random_state=RANDOM_SEED), {
        'n_estimators': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    'GradientBoosting': (GradientBoostingRegressor(random_state=RANDOM_SEED), {
        'n_estimators': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
    'ExtraTrees': (ExtraTreesRegressor(random_state=RANDOM_SEED), {
        'n_estimators': [500, 1000],
        'max_depth': [None, 10, 20]
    }),
    'RandomForest': (RandomForestRegressor(random_state=RANDOM_SEED), {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20]
    }),
    'AdaBoost': (AdaBoostRegressor(random_state=RANDOM_SEED), {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1]
    }),
    'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=RANDOM_SEED), {
        'max_iter': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'max_depth': [None, 10]
    }),
    'CatBoost': (CatBoostRegressor(verbose=False, random_state=RANDOM_SEED), {
        'iterations': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'depth': [3, 6]
    })
}

# 📌 Entrenamiento con GridSearchCV y medición de emisiones y tiempo
best_models = {}
results = {}

for model_name, (model, params) in models.items():
    print(f"\n🔍 Evaluando modelo: {model_name}")

    # Iniciar medición de tiempo y emisiones con `allow_multiple_runs=True`
    start_time = time.time()
    tracker = EmissionsTracker(log_level="error", allow_multiple_runs=True)
    tracker.start()

    if params:  # Si tiene hiperparámetros, usa GridSearchCV
        grid_search = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:  # Si no tiene hiperparámetros, entrena directamente
        best_model = model.fit(X_train, y_train)
        best_params = "N/A"

    # Evaluación en validación
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    std_rmse = np.std(np.sqrt(-cross_val_score(best_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)))
    r2 = r2_score(y_val, y_pred)

    # Medir tiempo y emisiones
    elapsed_time = time.time() - start_time
    emissions = tracker.stop() or 0.0  # Evitar error en NoneType

    # Guardar modelo entrenado
    joblib.dump(best_model, f"/mnt/data/best_{model_name}.pkl")

    # Guardar resultados
    best_models[model_name] = {"model": best_model, "RMSE": rmse, "Best Params": best_params}
    results[model_name] = {
        "RMSE": rmse, 
        "Std RMSE": std_rmse, 
        "R2": r2, 
        "Tiempo (s)": elapsed_time, 
        "Emisiones CO₂ (g)": emissions,
        "Mejores Hiperparámetros": best_params
    }

    print(f"✅ {model_name} - RMSE: {rmse:.4f} | Std: {std_rmse:.4f} | R2: {r2:.4f} | Tiempo: {elapsed_time:.2f}s | Emisiones: {emissions:.4f}g | Best Params: {best_params}")

# 📊 Convertir resultados en DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="RMSE")

# 📊 Guardar resultados en CSV
results_df.to_csv("/mnt/data/model_performance.csv", index=True)

# 📊 Mostrar resultados finales
print("\n📊 Resultados RMSE, Std, R2, Emisiones, Tiempo y Mejores Hiperparámetros:")
print(results_df)
print(f"\n🏆 Mejor modelo individual: {results_df.index[0]} con RMSE: {results_df.iloc[0]['RMSE']:.4f}")
print(f"📁 Evaluación completada. Resultados guardados en: /mnt/data/model_performance.csv")


# # Voting 

# In[3]:


import pandas as pd
import numpy as np
import joblib
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ================================================================
# 🔹 Cargar modelos entrenados
# ================================================================
models = {
    "CatBoost": joblib.load("/mnt/data/best_CatBoost.pkl"),
    "XGBoost": joblib.load("/mnt/data/best_XGBoost.pkl"),
    "GradientBoosting": joblib.load("/mnt/data/best_GradientBoosting.pkl"),
    "RandomForest": joblib.load("/mnt/data/best_RandomForest.pkl"),
}

# ================================================================
# 🔹 Cargar datos
# ================================================================
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = train_data.drop(columns=["Producto 1"])
y = train_data["Producto 1"]

# 🔹 Imputar valores faltantes
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🔹 Escalado
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# 🔹 Split en train y validación
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ================================================================
# 🔹 Definir modelos para el Voting Regressor
# ================================================================
ensemble_models = [(name, model) for name, model in models.items()]

# 🔹 Espacio de búsqueda para pesos
search_space = [Real(0.1, 10.0, name=f'weight_{i}') for i in range(len(ensemble_models))]

@use_named_args(search_space)
def objective(**params):
    weights = list(params.values())
    ensemble = VotingRegressor(estimators=ensemble_models, weights=weights)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# ================================================================
# 🔹 Optimización bayesiana de pesos
# ================================================================
print("🔍 Buscando pesos óptimos para Voting Regressor...")
opt_result = gp_minimize(objective, search_space, n_calls=40, random_state=42)
optimal_weights = opt_result.x
print(f"\n🎯 Pesos optimizados: {optimal_weights}")

# ================================================================
# 🔹 Entrenar Voting con pesos optimizados
# ================================================================
voting_model_optimized = VotingRegressor(estimators=ensemble_models, weights=optimal_weights)
voting_model_optimized.fit(X_train, y_train)

# ================================================================
# 🔹 Evaluar modelo Voting
# ================================================================
y_pred_voting = voting_model_optimized.predict(X_val)
metrics_voting = {
    "R2": r2_score(y_val, y_pred_voting),
    "RMSE": np.sqrt(mean_squared_error(y_val, y_pred_voting)),
    "MAE": mean_absolute_error(y_val, y_pred_voting)
}

print("\n📊 Voting Model Metrics (Optimizados):")
for metric, value in metrics_voting.items():
    print(f"🔹 {metric}: {value:.4f}")

# ================================================================
# 🔹 Guardar modelo y métricas
# ================================================================
joblib.dump(voting_model_optimized, "/mnt/data/best_voting_ensemble_recalibrated.pkl")
print("\n✅ Modelo Voting optimizado guardado en: /mnt/data/best_voting_ensemble_recalibrated.pkl")

metrics_df = pd.DataFrame([metrics_voting])
metrics_df.to_csv("/mnt/data/voting_ensemble_metrics.csv", index=False)
print("\n📁 Métricas guardadas en: /mnt/data/voting_ensemble_metrics.csv")

