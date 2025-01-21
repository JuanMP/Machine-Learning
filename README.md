# Proyectos de Machine Learning

Este repositorio contiene una colección de proyectos relacionados con el aprendizaje automático (Machine Learning), cada uno de ellos implementado con diferentes técnicas y modelos. Cada proyecto se encuentra en su propia subcarpeta, con su código, datos y documentación correspondiente. A medida que agregues más proyectos, este repositorio se expandirá para incluir más ejercicios y estudios de casos.

## Proyectos

### 1. **Proyecto 1: Análisis del perfil crediticio de clientes bancarios**

#### Descripción:
En este proyecto se desarrollan **dos modelos de aprendizaje automático** para predecir:

1. **Si se concederá un préstamo** a un nuevo cliente.
2. **El riesgo asociado** al cliente.

Este proyecto utiliza datos históricos de clientes de una entidad bancaria para entrenar los modelos y hacer predicciones sobre la probabilidad de aprobación de préstamos y el nivel de riesgo crediticio.

#### Tecnologías utilizadas:

- **Python**: Lenguaje de programación principal.
- **Jupyter Notebooks**: Entorno interactivo para desarrollar y ejecutar el código.
- **NumPy**: Manipulaciones numéricas y operaciones con arrays.
- **Pandas**: Manipulación y análisis de datos.
- **Matplotlib y Seaborn**: Visualización de datos y gráficos.
- **Optuna**: Optimización de hiperparámetros.
- **Imbalanced-learn (imblearn)**: Manejo de clases desequilibradas (SMOTE).
- **Scikit-learn**: Modelos de aprendizaje automático y evaluación:
  - **Clasificación**: `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`, `XGBClassifier`.
  - **Regresión**: `LinearRegression`, `RandomForestRegressor`, `DecisionTreeRegressor`.
  - **Evaluación de modelos**: `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `mean_squared_error`, `r2_score`.
  - **Preprocesamiento**: `StandardScaler`, `train_test_split`, `GridSearchCV`, `cross_val_score`.
- **XGBoost**: Modelo de clasificación utilizando Gradient Boosting.
- **Collections (Counter)**: Gestión y análisis de distribuciones de datos.

## Cómo ejecutar los proyectos

1. Clona el repositorio:
    ```bash
    git clone https://github.com/tu_usuario/MACHINE_LEARNING.git
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

3. Navega a la carpeta del proyecto y abre el archivo Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Ejecuta las celdas del notebook de manera secuencial para ver los resultados

## Contribuciones

Las contribuciones son bienvenidas. Si tienes sugerencias, correcciones o mejoras, por favor abre un "issue" o envía un "pull request"

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles

## Posibles extensiones futuras

- Optimización de modelos con más técnicas de ajuste de hiperparámetros
- Pruebas adicionales con otros modelos de clasificación y regresión
- Implementación de técnicas adicionales de manejo de desequilibrio en los datos
- Evaluación de la efectividad de los modelos con un conjunto de datos más grande
