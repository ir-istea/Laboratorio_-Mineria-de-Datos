# TelcoVision - Predicción de Churn con MLOps

Proyecto de predicción de abandono de clientes (churn) implementado con una arquitectura MLOps completa utilizando PyCaret, DVC y MLflow.

## Descripción del Proyecto

TelcoVision es un proyecto académico de Machine Learning Operations (MLOps) que predice el churn de clientes de una empresa de telecomunicaciones utilizando técnicas de AutoML y herramientas modernas de MLOps.

**Dataset**: 10,000 clientes con 12 variables predictoras  
**Variable objetivo**: Churn binario (0 = cliente activo, 1 = abandono)  
**Objetivo de accuracy**: 85%


## Stack Tecnológico

- **ML Framework**: PyCaret 3.x (AutoML)
- **Versionado de datos**: DVC
- **Tracking de experimentos**: MLflow 2.9.2
- **Repositorio**: GitHub + DagsHub
- **CI/CD**: GitHub Actions
- **Entorno**: Python 3.9 con conda

## Inicio Rápido

### Opción 1: Configuración Automática (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/ir-istea/Laboratorio_-Mineria-de-Datos.git
cd tp-labMineriaDeDatos/resolucion/telco

# 2. Ejecutar script de configuración
./setup_local_env.sh
```

### Opción 2: Configuración Manual

```bash
# 1. Crear entorno conda
conda create -n pycaret-env python=3.9 -y
conda activate pycaret-env

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales de DVC
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TU_USUARIO_DAGSHUB
dvc remote modify origin --local password TU_TOKEN_DAGSHUB

# 4. Crear archivo .env
cat > .env << EOF
MLFLOW_TRACKING_URI=https://dagshub.com/TU_USUARIO/Laboratorio_-Mineria-de-Datos.mlflow
MLFLOW_TRACKING_USERNAME=TU_USUARIO
MLFLOW_TRACKING_PASSWORD=TU_TOKEN
EOF

# 5. Descargar datos
dvc pull

# 6. Ejecutar pipeline
dvc repro
```


## Pipeline de Datos

El pipeline está definido en `dvc.yaml` y consta de 4 etapas:

### 1. Preparación de Datos (`data_prep`)
```bash
dvc repro data_prep
```
- Lee `data/raw/telco_churn.csv`
- Aplica limpieza básica
- Guarda `data/processed/telco_churn_processed.csv`

### 2. Entrenamiento (`train`)
```bash
dvc repro train
```
- Configura PyCaret con logging a MLflow
- Compara 5 modelos: Logistic Regression, Random Forest, XGBoost, Naive Bayes, SVM
- Optimiza hiperparámetros del mejor modelo
- Registra todos los experimentos en MLflow

### 3. Evaluación (`evaluate`)
```bash
dvc repro evaluate
```
- Busca el mejor modelo en MLflow
- Evalúa en el conjunto de prueba
- Registra métricas finales (accuracy, precision, recall, F1)
- Genera gráficos (Confusion Matrix, ROC Curve)

### 4. Promoción de Modelo (`promote_model`)
```bash
dvc repro promote_model
```
- Busca el modelo con mejor `final_accuracy` en MLflow
- Lo registra en el Model Registry
- Lo promueve a stage `Production`
- Archiva versiones anteriores

> [!IMPORTANT]
> El modelo promovido a `Production` es el que se usa automáticamente en el proyecto de despliegue (`resolucion/telco_prod`).

## Comandos Principales

### Gestión del Ambiente
```bash
conda activate pycaret-env      # Activar entorno
conda deactivate                # Desactivar entorno
./verify_env.sh                 # Verificar configuración
```

### Ejecución del Pipeline
```bash
dvc repro                       # Ejecutar pipeline completo
dvc repro train                 # Solo entrenamiento
dvc repro evaluate              # Solo evaluación
dvc status                      # Ver estado del pipeline
```

### Gestión de Datos con DVC
```bash
dvc pull                        # Descargar datos desde DagsHub
dvc push                        # Subir datos/modelos a DagsHub
dvc dag                         # Visualizar DAG del pipeline
```

### Gestión de Código con Git
```bash
git status                      # Ver cambios
git add .                       # Agregar cambios
git commit -m "mensaje"         # Hacer commit
git push                        # Subir a GitHub
```

## Parámetros del Modelo

Los parámetros se configuran en `params.yaml`:

```yaml
train_size: 0.8                 # Split 80/20 train/test
seed: 42                        # Semilla para reproducibilidad
metric: 'Accuracy'              # Métrica de ordenamiento
models_to_compare: ['lr', 'rf', 'xgboost', 'nb', 'svm']
target_metric: 0.85             # Objetivo de accuracy
```

Para modificar parámetros:
1. Edita `params.yaml`
2. Ejecuta `dvc repro`
3. DVC detectará el cambio y re-ejecutará las etapas necesarias

## Visualización de Resultados

### En DagsHub/MLflow

Todos los experimentos se registran automáticamente en:
https://dagshub.com/ignacio.rimasa/Laboratorio_-Mineria-de-Datos

**Pestaña "Experiments"**:
- Comparación de todos los modelos entrenados
- Métricas: Accuracy, AUC, Recall, Precision, F1
- Hiperparámetros de cada modelo
- Gráficos: matriz de confusión, curva ROC, feature importance
- Modelos descargables

### Tracking de Experimentos: Local vs. DagsHub

Puedes controlar dónde se registran tus experimentos de MLflow directamente desde el archivo `params.yaml`.

```yaml
# params.yaml
# ...
# MLflow configuration
track_to_dagshub: false # Poner en 'true' para registrar en DagsHub
dagshub_tracking_uri: "https://dagshub.com/ignacio.rimasa/Laboratorio_-Mineria-de-Datos.mlflow"
```

#### Para registrar en DagsHub (por defecto para CI/CD)

1.  Asegúrate de que la bandera esté en `true`:
    ```yaml
    track_to_dagshub: true
    ```
2.  Ejecuta el pipeline como de costumbre:
    ```bash
    dvc repro
    ```
    Los experimentos aparecerán en la pestaña "Experiments" de tu repositorio de DagsHub.

#### Para registrar Localmente

1.  Asegúrate de que la bandera esté en `false`:
    ```yaml
    track_to_dagshub: false
    ```
2.  En una terminal, inicia la interfaz de usuario de MLflow:
    ```bash
    mlflow ui
    ```
    Esto iniciará un servidor local en `http://127.0.0.1:5000` y creará un directorio `mlruns` para almacenar los experimentos.
3.  En una segunda terminal, ejecuta el pipeline:
    ```bash
    dvc repro
    ```
    Ahora puedes ver tus experimentos y modelos en tu navegador abriendo `http://127.0.0.1:5000`.

### Localmente

```bash
cat logs.log                    # Ver logs de ejecución
```

## Características Destacadas

### Integración PyCaret + MLflow

El entrenamiento utiliza la integración nativa de PyCaret con MLflow:

```python
# src/train.py
exp = ClassificationExperiment()
exp.setup(
    data=df,
    target='churn',
    log_experiment=True,             # Activa MLflow
    experiment_name="telco-churn-prediction"
)
```

Esto registra automáticamente:
- Todos los modelos comparados
- Hiperparámetros
- Métricas de entrenamiento
- Gráficos de evaluación
- Modelos serializados

### Evaluación Inteligente

El script de evaluación busca automáticamente el mejor modelo:

```python
# src/evaluate.py
best_run = mlflow.search_runs(
    experiment_names=["telco-churn-prediction"],
    order_by=["metrics.Accuracy DESC"],
    max_results=1
)
```

No hay dependencia de archivos locales, todo se obtiene dinámicamente desde MLflow.

## CI/CD con GitHub Actions

El workflow `.github/workflows/ci.yaml` se ejecuta automáticamente en:
- Pull Requests a `main`
- Push a `main`

**Pasos del pipeline de CI/CD**:
1. Setup de Python 3.9
2. Instalación de dependencias
3. Configuración de DVC con credenciales
4. Descarga de datos (`dvc pull`)
5. Ejecución del pipeline (`dvc repro`)

## Solución de Problemas Comunes

### Error: "Conda not found"
Instala Miniconda: https://docs.conda.io/en/latest/miniconda.html

### Error: "401 Unauthorized" en dvc pull
Verifica tus credenciales de DagsHub:
```bash
dvc remote modify origin --local user TU_USUARIO
dvc remote modify origin --local password TU_TOKEN_CORRECTO
```

### Error: Importación de PyCaret falla
Reinstala PyCaret:
```bash
conda activate pycaret-env
pip install --upgrade pycaret[full]
```

### Error: "AttributeError: 'ThreadLocalVariable'"
Este error está resuelto en `requirements.txt` con `mlflow==2.9.2`.
Si persiste, verifica tu versión de MLflow:
```bash
pip install mlflow==2.9.2 --force-reinstall
```

## Obtener Token de DagsHub

1. Ve a https://dagshub.com/user/settings/tokens
2. Click en "New Token"
3. Dale un nombre descriptivo (ej: "local-dev")
4. Marca los permisos necesarios
5. Click en "Generate Token"
6. Copia el token (no podrás verlo de nuevo)

## Reproducibilidad

El proyecto garantiza reproducibilidad mediante:

- **Semillas fijas**: `seed: 42` en params.yaml
- **Hashes MD5**: DVC rastrea hashes de archivos en `dvc.lock`
- **Versiones fijas**: `mlflow==2.9.2` en requirements.txt
- **Entorno aislado**: Conda con Python 3.9
- **Versionado de datos**: DVC con remote en DagsHub

## Métricas Objetivo

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **Accuracy** | ≥ 85% | Porcentaje de predicciones correctas |
| **Precision** | ≥ 80% | De los predichos como churn, % reales |
| **Recall** | ≥ 75% | De los churn reales, % detectados |
| **F1-Score** | ≥ 77% | Media armónica de precision y recall |

## Documentación Adicional

- **SETUP_GUIDE.md**: Guía detallada de configuración
- **historial-15102025-conMlflow-v3.md**: Proceso de integración de MLflow
- **historial-25102025-conMlflow-v1.md**: Análisis de cumplimiento del proyecto
- **Notebooks**: Análisis exploratorio y feature engineering

## Guía de Reproducción Completa


### Ejecución Rápida (Local → DagsHub)

```bash
# 1. Configurar entorno
cd resolucion/telco
conda activate pycaret-env

# 2. Asegurar tracking remoto en params.yaml
# track_to_dagshub: true

# 3. Ejecutar pipeline completo
dvc repro
```

Esto entrenará, evaluará y promoverá automáticamente el mejor modelo a `Production` en MLflow.

### Verificar Modelo en DagsHub

Después de ejecutar el pipeline, ve a:
- **Experiments**: https://dagshub.com/ignacio.rimasa/Laboratorio_-Mineria-de-Datos/experiments
- **Models**: Verifica que `telco-churn-prediction` esté en stage `Production`

## Tiempo de Ejecución

- **Setup inicial**: 10-15 minutos
- **Pipeline completo** (`dvc repro`): 5-15 minutos
  - `data_prep`: ~5 segundos
  - `train`: ~5-15 minutos (depende del hardware)
  - `evaluate`: ~10 segundos

## Contribuciones

Este es un proyecto académico. Para contribuir:

1. Fork del repositorio
2. Crea una rama con tu feature (`git checkout -b feature/nueva-feature`)
3. Commit de tus cambios (`git commit -m 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

