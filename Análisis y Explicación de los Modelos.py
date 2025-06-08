#!/usr/bin/env python
# coding: utf-8

# # Análisis y Explicación de Modelos de Regresión
# 
# Este notebook analiza en detalle cada uno de los modelos que componen nuestro ensamble final, utilizando técnicas modernas de XAI (Explainable AI)
# 
# Contenido
# 
# Configuración inicial y carga de datos
# 
# Análisis de características e importancia
# 
# Análisis individual de modelos
# 
# Análisis del ensamble
# 
# Interpretación local y global de predicciones

# In[62]:


# Importar librerías necesarias
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# Librerías para explicabilidad
import shap
from sklearn.inspection import (
    permutation_importance,
    partial_dependence,
    PartialDependenceDisplay
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# Configuración de visualización
sns.set_theme(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['figure.dpi'] = 100


# In[63]:


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data_and_models():
    # 📂 Cargar datos
    data = pd.read_csv("/mnt/data/train_data_clean.csv")


    # de data eliminamos la columna Producto 2
    data = data.drop('Producto 2', axis=1)

    # 🛠️ Eliminar columnas con correlación NaN con 'Producto 1'
    corr = data.corr()['Producto 1']
    columns_to_drop = corr[corr.isna()].index.tolist()
    
    # Guardar las columnas eliminadas
    with open('/mnt/data/columns_to_drop.txt', 'w') as f:
        for column in columns_to_drop:
            f.write(column + '\n')

    data = data.drop(columns_to_drop, axis=1)

    # 📊 Preparación de datos
    X = data.drop('Producto 1', axis=1)
    y = data['Producto 1']

    # Guardar `X.csv` sin la columna objetivo
    X.to_csv('/mnt/data/X.csv', index=False)

    # 🛠️ Manejo de valores NaN: reemplazar por la media
    X = X.fillna(X.mean())

    # 🔍 Escalar datos con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 📌 Cargar **solo los modelos usados en Blending**
    models = {
        'CatBoost': joblib.load('/mnt/data/best_CatBoost.pkl'),
        'XGBoost': joblib.load('/mnt/data/best_XGBoost.pkl'),
        'AdaBoost': joblib.load('/mnt/data/best_AdaBoost.pkl'),
        'GradientBoosting': joblib.load('/mnt/data/best_GradientBoosting.pkl'),
        'RandomForest': joblib.load('/mnt/data/best_RandomForest.pkl')
    }

    print("\n✅ Datos y modelos cargados correctamente.")

    return X, y, models

# 🔹 Ejecutar la función y cargar los datos y modelos
X, y, models = load_data_and_models()


# ## Análisis de Características

# In[64]:


def analyze_feature_distributions(X, y):
    #Analizar distribuciones de características y correlaciones\\\
    # Correlación con la variable objetivo
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [X[col].corr(y) for col in X.columns]
    }).sort_values('correlation', key=abs, ascending=False)
    
    # Top 10 correlaciones
    plt.figure(figsize=(12, 6))
    sns.barplot(data=correlations.head(10), x='correlation', y='feature')
    plt.title('Top 10 Correlaciones con la Variable Objetivo')
    plt.show()
    
    # Matriz de correlación para las top 15 características
    top_features = correlations['feature'].head(15).tolist()
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        X[top_features].corr(),
        annot=True,
        cmap='coolwarm',
        center=0
    )
    plt.title('Matriz de Correlación - Top 15 Características')
    plt.show()
    
    return correlations
    
correlations = analyze_feature_distributions(X, y)


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Función para calcular métricas de rendimiento
def calculate_model_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'R2': r2}

# 🔹 Análisis de Partial Dependence Plots (PDP)
def analyze_unified_partial_dependence(models, X, features):
    n_features = len(features)
    n_models = len(models)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
    if n_features == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, n_models))

    for idx, feature in enumerate(features):
        ax = axes[idx]

        for (model_name, model), color in zip(models.items(), colors):
            if hasattr(model, "feature_names_in_") and feature not in model.feature_names_in_:
                continue  # Evitar errores en modelos sin la característica

            try:
                display = PartialDependenceDisplay.from_estimator(
                    model, X, [feature], ax=ax, random_state=42, subsample=1000
                )
                
                # Modificar líneas para mejorar visualización
                display.lines_[0][0].set_label(model_name)
                display.lines_[0][0].set_color(color)

                if hasattr(display, 'lines_individual_') and display.lines_individual_:
                    for ice_line in display.lines_individual_[0]:
                        ice_line.set_color(color)
                        ice_line.set_alpha(0.1)

            except Exception as e:
                print(f"⚠️ PDP no disponible para {model_name} en {feature}: {e}")

        ax.set_title(f'Partial Dependence Plot - {feature}')
        ax.legend()
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial dependence')

    plt.tight_layout()
    plt.show()

# 🔹 Análisis de SHAP (Feature Importance)
def plot_unified_shap_analysis(models, X):
    shap.initjs()
    feature_importance_dict = {}

    for model_name, model in models.items():
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_dict[model_name] = feature_importance
        except Exception as e:
            print(f"⚠️ SHAP no disponible para {model_name}: {e}")

    if not feature_importance_dict:
        print("⚠️ Ningún modelo generó valores SHAP.")
        return None

    importance_df = pd.DataFrame(feature_importance_dict, index=X.columns)
    importance_df['mean'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('mean', ascending=True).tail(15)

    # 🔹 Visualización con heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(importance_df.drop('mean', axis=1).T, cmap='viridis', cbar_kws={'label': 'SHAP Value Magnitude'})
    plt.title('SHAP Feature Importance Heatmap')
    plt.tight_layout()
    plt.show()

    return importance_df

# 🔹 Análisis de Permutation Importance
def analyze_unified_permutation_importance(models, X, y):
    all_importance_dfs = []

    for model_name, model in models.items():
        try:
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': result.importances_mean,
                'std': result.importances_std,
                'model': model_name
            })
            all_importance_dfs.append(importance_df)
        except Exception as e:
            print(f"⚠️ Permutation Importance no disponible para {model_name}: {e}")

    if not all_importance_dfs:
        print("⚠️ Ningún modelo generó Permutation Importance.")
        return None

    combined_df = pd.concat(all_importance_dfs)
    top_features = combined_df.groupby('feature')['importance'].mean().nlargest(15).index
    plot_df = combined_df[combined_df['feature'].isin(top_features)]

    # 🔹 Visualización con barplot
    plt.figure(figsize=(15, 8))
    sns.barplot(data=plot_df, x='importance', y='feature', hue='model', palette='viridis')
    plt.title('Unified Permutation Importance - Top 15 Características')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return combined_df

# 🔹 Análisis de todos los modelos
def analyze_all_models(models, X, y):
    metrics_dict = {}

    for model_name, model in models.items():
        try:
            y_pred = model.predict(X)
            metrics_dict[model_name] = calculate_model_metrics(y, y_pred)
        except Exception as e:
            print(f"⚠️ Error en predicción para {model_name}: {e}")

    # 📊 Mostrar métricas
    if metrics_dict:
        metrics_df = pd.DataFrame(metrics_dict).T.round(4)
        print("\n📊 Comparación de métricas entre modelos:")
        print(metrics_df)
    else:
        print("⚠️ No se calcularon métricas.")

    # 📊 Analizar importancia permutada
    print("\n🔹 Analizando importancia de características unificada...")
    importance_df = analyze_unified_permutation_importance(models, X, y)

    # 📊 Analizar SHAP
    print("\n🔹 Realizando análisis SHAP unificado...")
    shap_importance_df = plot_unified_shap_analysis(models, X)

    return metrics_df, importance_df, shap_importance_df

# ================================================================
# 🔹 Cargar modelos y datos preprocesados
# ================================================================
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 📂 Cargar modelos utilizados en el Blending
models = {
    'CatBoost': joblib.load('/mnt/data/best_CatBoost.pkl'),
    'XGBoost': joblib.load('/mnt/data/best_XGBoost.pkl'),
    'AdaBoost': joblib.load('/mnt/data/best_AdaBoost.pkl'),
    'GradientBoosting': joblib.load('/mnt/data/best_GradientBoosting.pkl'),
    'RandomForest': joblib.load('/mnt/data/best_RandomForest.pkl'),
}

# 📂 Cargar datos
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = train_data.drop(columns=["Producto 1"])
y = train_data["Producto 1"]

# 🛠️ Imputación y escalado
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# 📊 Asegurar consistencia en columnas
for model_name, model in models.items():
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        X_scaled = X_scaled[expected_features]

# ================================================================
# 🔹 Ejecutar el análisis unificado
# ================================================================
results = analyze_all_models(models, X_scaled, y)


# In[66]:


import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import pandas as pd

# 🔹 Función para graficar análisis SHAP unificado
def plot_unified_shap_complete_analysis(models, X, figsize=(30, 30)):
    """
    Crea una visualización unificada del análisis SHAP completo para múltiples modelos,
    incluyendo tanto el summary plot como el feature importance plot.
    
    Parameters:
    -----------
    models : dict
        Diccionario con los modelos {'nombre_modelo': modelo}
    X : pd.DataFrame
        DataFrame con las características
    figsize : tuple
        Tamaño de la figura (ancho, alto)
    """
    shap.initjs()

    # 🔹 Filtrar modelos compatibles con SHAP
    compatible_models = {
        name: model for name, model in models.items()
        if hasattr(model, "predict") and hasattr(model, "fit")  
    }

    # 🔹 Número de modelos compatibles
    n_models = len(compatible_models)
    if n_models == 0:
        print("⚠️ No hay modelos compatibles con SHAP en el conjunto de modelos.")
        return None

    # 🔹 Crear subplots dinámicos según el número de modelos
    fig, axes = plt.subplots(n_models, 2, figsize=figsize)

    # 🔹 Calcular y plotear valores SHAP para cada modelo
    for idx, (model_name, model) in enumerate(compatible_models.items()):
        try:
            # Crear explicador SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # 🔹 Gráfico de Summary Plot (SHAP Values)
            plt.sca(axes[idx, 0])
            shap.summary_plot(shap_values, X, show=False, plot_size=None)
            axes[idx, 0].set_title(f'SHAP Summary Plot - {model_name}')

            # 🔹 Gráfico de Feature Importance (Barras)
            plt.sca(axes[idx, 1])
            shap.summary_plot(shap_values, X, plot_type="bar", show=False, plot_size=None)
            axes[idx, 1].set_title(f'SHAP Feature Importance - {model_name}')

        except Exception as e:
            print(f"⚠️ No se pudo calcular SHAP para {model_name}: {e}")

    # 🔹 Ajustar el layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Unified Complete SHAP Analysis for All Models', fontsize=16, y=0.98)

    return plt.gcf()

# ================================================================
# 🔹 Cargar modelos utilizados en el Voting
# ================================================================
models = {
    'CatBoost': joblib.load('/mnt/data/best_CatBoost.pkl'),
    'XGBoost': joblib.load('/mnt/data/best_XGBoost.pkl'),
    'AdaBoost': joblib.load('/mnt/data/best_AdaBoost.pkl'),
    'GradientBoosting': joblib.load('/mnt/data/best_GradientBoosting.pkl'),
    'RandomForest': joblib.load('/mnt/data/best_RandomForest.pkl'),  
}

# 📂 Cargar datos
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = train_data.drop(columns=["Producto 1"])

# 🔹 Manejo de valores faltantes e imputación
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🔹 Escalar datos con StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# 🔹 Asegurar consistencia en las columnas para cada modelo
for model_name, model in models.items():
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        X_scaled = X_scaled[expected_features]

# ================================================================
# 🔹 Ejecutar la visualización SHAP
# ================================================================
fig = plot_unified_shap_complete_analysis(models, X_scaled)

# 📁 Guardar la figura
if fig:
    fig.savefig("/mnt/data/unified_shap_analysis.png", dpi=600)
    print("\n✅ Análisis SHAP guardado en: /mnt/data/unified_shap_analysis.png")

# 📊 Mostrar la figura si existe
if fig:
    plt.show()


# # Análisis del Voting

# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def plot_voting_analysis(voting_model, X, y, feature_names=None, figsize=(20, 12)):
    """
    Crea un análisis detallado del modelo Voting Regressor.
    
    📌 Se incluyen:
    - Importancia de características basada en los pesos del Voting
    - Correlación entre predicciones de los modelos base
    - Análisis de residuos y distribución
    
    Parameters:
    -----------
    voting_model : VotingRegressor
        Modelo Voting Regressor optimizado
    X : pd.DataFrame
        DataFrame con las características
    y : array-like
        Variable objetivo
    feature_names : list, optional
        Nombres de las características (se infiere de X si es None)
    figsize : tuple
        Tamaño de la figura
    """
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]

    models = voting_model.estimators_  # Modelos base en el Voting
    model_names = [name for name, _ in voting_model.estimators]
    weights = np.array(voting_model.weights)  # Pesos asignados a cada modelo
    
    # 📌 Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 🔹 1. Importancia de Características en el Voting
    importances = []
    for model_name, model in zip(model_names, models):
        if hasattr(model, 'feature_importances_'):
            importances.append(pd.Series(model.feature_importances_ * weights[model_names.index(model_name)], index=feature_names))

    if importances:
        voting_importance = pd.concat(importances, axis=1).sum(axis=1).sort_values(ascending=True)

        # 📊 Gráfico de Importancia Basado en Pesos del Voting
        ax = axes[0, 0]
        voting_importance.plot(kind='barh', ax=ax)
        ax.set_title('📊 Voting Feature Importance')
        ax.set_xlabel('Weighted Importance')

    # 🔹 2. Análisis de Predicciones Individuales vs Voting
    individual_preds = []
    for model_name, model in zip(model_names, models):
        try:
            individual_preds.append(pd.Series(model.predict(X), name=model_name))
        except Exception as e:
            print(f"⚠️ Error al predecir con {model_name}: {e}")

    predictions_df = pd.concat(individual_preds, axis=1)

    # 📌 **Aplicar los pesos correctamente**
    weights /= weights.sum()  # Normalización de pesos
    y_pred_voting = np.dot(predictions_df.values, weights)

    predictions_df['Voting'] = y_pred_voting
    predictions_df['True'] = y

    # 📊 Mapa de Calor de Correlaciones
    ax = axes[0, 1]
    sns.heatmap(predictions_df.corr(), annot=True, cmap='viridis', ax=ax)
    ax.set_title('📊 Correlation between Model Predictions')

    # 🔹 3. Análisis de Residuos
    residuals = y - y_pred_voting

    # 📊 Scatter Plot de Residuos vs Predicciones
    ax = axes[1, 0]
    sns.scatterplot(x=y_pred_voting, y=residuals, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('📊 Residuals vs Predicted Values')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')

    # 📊 Histograma de Residuos
    ax = axes[1, 1]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title('📊 Distribution of Residuals')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('📊 Voting Ensemble Model Analysis', fontsize=16, y=0.98)

    # 📌 **Calcular y mostrar métricas**
    metrics = {
        'R2': r2_score(y, y_pred_voting),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred_voting)),
        'MAE': mean_absolute_error(y, y_pred_voting)
    }

    print("\n📊 Voting Model Metrics:")
    for metric, value in metrics.items():
        print(f"🔹 {metric}: {value:.4f}")

    return {
        'predictions': predictions_df,
        'residuals': residuals,
        'metrics': metrics,
        'voting_importance': voting_importance if importances else None
    }

# ================================================================
# 🔹 Cargar el Modelo Voting Optimizado
# ================================================================
voting_model = joblib.load('/mnt/data/best_voting_ensemble_recalibrated.pkl')

# 📂 Cargar datos
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = train_data.drop(columns=["Producto 1"])
y = train_data["Producto 1"]

# 🔹 Manejo de valores faltantes e imputación
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🔹 Escalar datos con StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ================================================================
# 🔹 Ejecutar el Análisis del Voting
# ================================================================
results = plot_voting_analysis(voting_model, X_scaled, y)

# 📊 Mostrar la figura
plt.show()

# 📁 Guardar resultados
predictions_df = results['predictions']
predictions_df.to_csv('/mnt/data/voting_predictions.csv', index=False)

residuals_df = pd.DataFrame({'Residuals': results['residuals']})
residuals_df.to_csv('/mnt/data/voting_residuals.csv', index=False)

metrics_df = pd.DataFrame([results['metrics']])
metrics_df.to_csv('/mnt/data/voting_metrics.csv', index=False)

if results['voting_importance'] is not None:
    results['voting_importance'].to_csv('/mnt/data/voting_feature_importance.csv', index=True)

print("\n✅ Todos los resultados del análisis del Voting han sido guardados.")


# In[80]:


import pandas as pd
import matplotlib.pyplot as plt
import joblib

def analyze_ensemble_contributions(voting_model):
    """
    📊 Analiza y visualiza las contribuciones (pesos) de cada modelo base en el Voting Regressor.
    """

    # 🔹 Obtener nombres de modelos y pesos
    estimators = [name for name, _ in voting_model.estimators]
    weights = np.array(voting_model.weights)

    # 🔹 Gráfico de barras de pesos
    plt.figure(figsize=(10, 6))
    bars = plt.bar(estimators, weights)
    plt.title('📊 Pesos de los Modelos en el Voting Regressor')
    plt.ylabel("Peso Asignado")
    plt.xticks(rotation=45)

    # 🔹 Etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    # 🔹 Calcular contribución relativa en porcentaje
    relative_contribution = (weights / weights.sum()) * 100
    contribution_df = pd.DataFrame({
        'Modelo': estimators,
        'Peso': weights,
        'Contribución (%)': relative_contribution
    })

    print("\n📌 Contribución de cada modelo al Voting Regressor:")
    print(contribution_df.to_string(index=False))

    return contribution_df

# ================================================================
# 🔹 Cargar modelo Voting optimizado
# ================================================================
voting_model = joblib.load("/mnt/data/best_voting_ensemble_recalibrated.pkl")

# 🔹 Ejecutar análisis de contribuciones
ensemble_contributions = analyze_ensemble_contributions(voting_model)


# ## Análisis de Modelos Individuales

# In[76]:


import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================================================================
# 🔹 Funciones auxiliares
# ================================================================

def calculate_model_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def analyze_permutation_importance(model, X, y, model_name):
    try:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title(f'📊 Permutation Importance - {model_name}')
        plt.show()

        return importance_df
    except Exception as e:
        print(f"⚠️ Error en Permutation Importance para {model_name}: {e}")
        return None

def plot_shap_analysis(model, X, model_name):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type='bar')
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.show()
        return shap_values
    except Exception as e:
        print(f"⚠️ Error al calcular SHAP para {model_name}: {e}")
        return None

def analyze_partial_dependence(model, X, features, model_name):
    try:
        display = PartialDependenceDisplay.from_estimator(
            model, X, features, kind='both', subsample=1000, random_state=42
        )
        plt.suptitle(f'Partial Dependence - {model_name}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Error en PDP para {model_name}: {e}")

def analyze_single_model(model, X, y, model_name):
    print("="*60)
    print(f'📊 Análisis del modelo: {model_name}')
    print("="*60)

    # 🔹 Predicciones y métricas
    y_pred = model.predict(X)
    metrics = calculate_model_metrics(y, y_pred)
    print("\n✅ Métricas de rendimiento:")
    for metric, value in metrics.items():
        print(f"🔹 {metric}: {value:.4f}")

    # 🔹 Permutation Importance
    print("\n📌 Analizando Permutation Importance...")
    importance_df = analyze_permutation_importance(model, X, y, model_name)

    # 🔹 SHAP
    print("\n📌 Analizando SHAP...")
    shap_values = plot_shap_analysis(model, X, model_name)

    # 🔹 Partial Dependence
    if importance_df is not None:
        top_features = importance_df['feature'].head(3).tolist()
        print(f"\n📌 Partial Dependence para: {top_features}")
        analyze_partial_dependence(model, X, top_features, model_name)

    return metrics, importance_df, shap_values

# ================================================================
# 🔹 Cargar modelos
# ================================================================
models = {
    "CatBoost": joblib.load("/mnt/data/best_CatBoost.pkl"),
    "XGBoost": joblib.load("/mnt/data/best_XGBoost.pkl"),
    "GradientBoosting": joblib.load("/mnt/data/best_GradientBoosting.pkl"),
    "RandomForest": joblib.load("/mnt/data/best_RandomForest.pkl"),
    "Voting": joblib.load("/mnt/data/best_voting_ensemble_recalibrated.pkl")  # Voting Regressor con pesos optimizados
}

# ================================================================
# 🔹 Cargar y preparar datos
# ================================================================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = data.drop(columns=["Producto 1"])
y = data["Producto 1"]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ================================================================
# 🔹 Ejecutar análisis para todos los modelos
# ================================================================
results = {}
for model_name, model in models.items():
    results[model_name] = analyze_single_model(model, X_scaled, y, model_name)

print("\n✅ Análisis completado para todos los modelos.")


# ## Análisis de Casos Específicos

# In[79]:


import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# ================================================================
# 🔹 Función para hacer predicciones con el Voting Regressor
# ================================================================
def voting_predict(voting_model, X):
    """
    📌 Genera predicciones del Voting Regressor y de cada modelo base.
    """
    models = voting_model.estimators_
    model_names = [name for name, _ in voting_model.estimators]
    
    # 📌 Obtener predicciones individuales de cada modelo base
    individual_preds = np.column_stack([model.predict(X) for model in models])
    
    # 📌 Predicción final del Voting (ponderada por los pesos)
    weights = np.array(voting_model.weights)
    weights /= weights.sum()  # Normalización de pesos
    voting_predictions = np.dot(individual_preds, weights)
    
    return voting_predictions, individual_preds, model_names

# ================================================================
# 🔹 Función para Analizar Casos Específicos en el Voting
# ================================================================
def analyze_specific_cases(voting_model, X, y, n_cases=5):
    """
    📊 Analizar predicciones específicas del Voting Regressor para entender su comportamiento.
    
    📌 Muestra:
      - Casos con mejor y peor predicción
      - Comparación de predicciones individuales vs Voting
      - Análisis SHAP en modelos base

    Parameters:
    -----------
    voting_model : VotingRegressor
        Modelo Voting Regressor optimizado
    X : pd.DataFrame
        DataFrame con características de entrada
    y : pd.Series
        Valores reales
    n_cases : int, optional
        Número de casos a analizar (por defecto 5)
    """
    
    # 📌 Obtener predicciones del Voting y modelos base
    predictions, individual_preds, model_names = voting_predict(voting_model, X)
    errors = np.abs(predictions - y)

    # 📌 Obtener índices de mejores y peores predicciones
    best_indices = np.argsort(errors)[:n_cases]
    worst_indices = np.argsort(errors)[-n_cases:]

    # 🔹 Función para analizar un caso específico
    def analyze_case(idx, case_type):
        print(f"\n{case_type} - Error: {errors[idx]:.4f}")
        print(f"Valor Real: {y.iloc[idx]:.4f}")
        print(f"Predicción Voting: {predictions[idx]:.4f}")

        # 📌 Obtener predicciones individuales de cada modelo base
        individual_predictions = individual_preds[idx]
        
        pred_df = pd.DataFrame({
            'Modelo': model_names,
            'Predicción': individual_predictions,
            'Diferencia con real': individual_predictions - y.iloc[idx]
        }).sort_values('Diferencia con real')

        print("\n📊 Predicciones individuales:")
        print(pred_df.to_string(index=False))

        # 🔹 SHAP Values para cada modelo individual
        for model_name, model in zip(model_names, voting_model.estimators_):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X.iloc[idx:idx+1])

                plt.figure(figsize=(10, 5))
                shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[idx],
                                matplotlib=True, show=False)
                plt.title(f'SHAP Force Plot - {case_type} - Modelo: {model_name}')
                plt.show()
            except Exception as e:
                print(f"⚠️ No se pudo calcular SHAP para {model_name}: {e}")

    # 📊 Analizar mejores casos
    print("\n=== 📈 Mejores Predicciones ===")
    for idx in best_indices:
        analyze_case(idx, f'Mejor Caso #{list(best_indices).index(idx)+1}')

    # 📊 Analizar peores casos
    print("\n=== 📉 Peores Predicciones ===")
    for idx in worst_indices:
        analyze_case(idx, f'Peor Caso #{list(worst_indices).index(idx)+1}')

# ================================================================
# 🔹 Cargar el Modelo Voting Regressor Recalibrado
# ================================================================
voting_model = joblib.load('/mnt/data/best_voting_ensemble_recalibrated.pkl')

# 📂 Cargar datos
train_data = pd.read_csv("/mnt/data/train_data_clean.csv")
X = train_data.drop(columns=["Producto 1"])
y = train_data["Producto 1"]

# 🔹 Manejo de valores faltantes e imputación
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 🔹 Escalar datos con StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ================================================================
# 🔹 Analizar Casos Específicos en el Voting
# ================================================================
analyze_specific_cases(voting_model, X_scaled, y)

