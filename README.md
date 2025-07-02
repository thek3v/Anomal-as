
# Proyecto de Detección de Anomalías: Riesgo de Crédito y Fraude

Este notebook implementa un pipeline completo de detección de anomalías en solicitudes de crédito, abordando el problema como una **clasificación multiclase desbalanceada**. Las clases objetivo son: `performing`, `credit_risk` y `fraud_risk`.

---

##  1. Carga de datos

- Cargamos un archivo `.pkl` con estructura tipo diccionario (ID → diccionario con info del crédito).
- Convertimos la información anidada (`merchant`, `country`, `checkout`, `financing_plan`) en columnas planas usando `json_normalize`.
- Unimos todo en un `DataFrame` tabular llamado `df_final`.

---

##  2. Preprocesamiento

- Exploramos la variable objetivo `target`, identificando un fuerte desbalance:
  - `performing` ≈ 91.7%
  - `credit_risk` ≈ 7.5%
  - `fraud_risk` ≈ 0.7%
- Eliminamos columnas irrelevantes o duplicadas.
- Creamos nuevas variables temporales (`created_month`, `day`, `hour`, `weekday`).
- Convertimos todas las variables categóricas con:
  - `LabelEncoder` (para columnas con pocos niveles)
  - `TargetEncoder` (con validación cruzada, para columnas de alta cardinalidad)

---

##  3. Feature Engineering

Aplicamos ingeniería de variables **postcodificación**, directamente sobre `X_train_te` y `X_test_te`:

- Variables derivadas de comportamiento financiero:
  - `amount_per_instalment`
  - `interest_ratio`
  - `downpayment_ratio`
  - `interest_rate_x_duration`

- Variables temporales:
  - `is_night_time`
  - `is_weekend`
  - `is_end_of_month`

- Variables tecnológicas:
  - `tech_complexity` (suma de codificaciones de modelo, SO y navegador)
  - `mobile_and_touch` (uso de móvil + pantalla táctil)

---

##  4. Manejo del desbalance

Se probaron tres enfoques para mejorar la detección de clases minoritarias:

1. **Solo `class_weight='balanced'`**
2. **Solo `SMOTE` parcial** (sin igualar completamente las clases)
3. **SMOTE + `class_weight='balanced'`** ✅ Mejor opción en nuestro caso

> Se usó `sampling_strategy={'credit_risk': 6000, 'fraud_risk': 1600}` para aumentar visibilidad sin perder realismo.

---

##  5. Modelos probados

Se entrenaron varios modelos para comparar rendimiento multiclase:

- `RandomForestClassifier` ✅
- `LogisticRegression` (multinomial) ✅
- `XGBoostClassifier`
- `CatBoostClassifier`
- `HistGradientBoostingClassifier`
- `BalancedBaggingClassifier`

Se evaluaron mediante:
- `classification_report` (precision, recall, f1 por clase)
- `confusion_matrix`
- `f1_macro`, `recall_fraud`, `recall_credit_risk`
- (opcional) visualización con UMAP o PCA

---

##  6. Guardado de artefactos

- Modelos entrenados se guardaron con `joblib` para reutilización:
  - `modelo_rf.pkl`, `modelo_logreg.pkl`, etc.
- Datos procesados guardados como:
  - `X_train_te.csv`, `X_test_te.csv`, `df_final.csv`
- El notebook completo exportado como `HTML` con:
  ```bash
  jupyter nbconvert --to html 1_anomalias.ipynb --output-dir=<ruta>

