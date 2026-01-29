Descripción del proyecto:
Proyecto de Machine Learning para predecir si un cliente bancario suscribirá un depósito a plazo (variable objetivo: deposit). El trabajo sigue las instrucciones de la asignatura Machine Learning for Business Decision Making (Assignment #1: Predicting Bank Product Subscription, 2025‑26): EDA simplificado, definición de esquemas de evaluación (holdout interno/externo), comparación de métodos básicos (Dummy, KNN, árboles), experimentos con métodos avanzados (Random Forest / Gradient Boosting / CatBoost), optimización de hiperparámetros, calibración de probabilidades y generación del modelo final para un conjunto de competición.

Contenido del repositorio:
`ASSIGNMENT__1.ipynb`
Notebook principal del trabajo. Contiene todo el flujo del experimento:

- Carga y preprocesado de los datos (`bank_23.pkl`).

- EDA y análisis de variables (numéricas y categóricas).

- Definición de particiones train/test y esquema de evaluación.

- Entrenamiento y comparación de modelos (básicos y avanzados) con búsqueda de hiperparámetros.

- Selección del modelo final, evaluación en test y obtención de predicciones para el conjunto de competición.

`bank_23.pkl`
Dataset principal en formato pickle. Es el conjunto de datos asignado al grupo, con información acerca de los clientes bancarios. Se utiliza para todo el entrenamiento, tuning y evaluación interna del modelo.

`bank_competition.pkl`
Conjunto de datos de competición proporcionado por la asignatura. No incluye la variable objetivo; se usa únicamente para generar las predicciones finales con el mejor modelo entrenado.

`final_model_catboost_pipeline.joblib`
Pipeline de preprocesado + modelo final entrenado. Permite cargar directamente el modelo y producir predicciones a partir de nuevas observaciones con la misma estructura que `bank_23.pkl`/`bank_competition.pkl`.

`LoadModel.ipynb`
Notebook auxiliar que muestra cómo cargar `final_model_catboost_pipeline.joblib` y aplicar el pipeline al conjunto de competición (u otros datos) para obtener predicciones. Es el punto de entrada recomendado si solo se quiere reutilizar el modelo final.

`predictions_bank_competition.csv`
Archivo con las predicciones finales del modelo sobre `bank_competition.pkl`.

`requirements.txt`
Lista mínima de dependencias de Python necesarias para ejecutar los notebooks y cargar el modelo
