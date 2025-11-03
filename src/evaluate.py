# src/evaluate.py (con MLflow)
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv

def evaluate_model():
    # Cargar parámetros PRIMERO
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # --- MLflow setup ---
    if params.get('track_to_dagshub', False):
        # Si vamos a DagsHub, cargar credenciales y configurar URI
        load_dotenv()
        mlflow.set_tracking_uri(params['dagshub_tracking_uri'])
    else:
        # Si es local, asegurarse de que no haya URI de tracking de DagsHub
        if 'MLFLOW_TRACKING_URI' in os.environ:
            del os.environ['MLFLOW_TRACKING_URI']
        if 'MLFLOW_TRACKING_USERNAME' in os.environ:
            del os.environ['MLFLOW_TRACKING_USERNAME']
        if 'MLFLOW_TRACKING_PASSWORD' in os.environ:
            del os.environ['MLFLOW_TRACKING_PASSWORD']

    experiment_name = "telco-churn-prediction"

    # Cargar datos de prueba (el mismo hold-out set que usó PyCaret)
    df = pd.read_csv(params['data_read_csv'])
    
    data_unseen = df.sample(frac=1-params['train_size'], random_state=params['seed'])
    X_unseen = data_unseen.drop('churn', axis=1)
    y_unseen = data_unseen['churn']

    # Buscar el mejor run dentro del experimento, ordenado por Accuracy
    best_run = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.Accuracy DESC"],
        max_results=1
    ).iloc[0]

    best_run_id = best_run["run_id"]
    print(f"Mejor Run ID encontrado: {best_run_id}")
    print(f"Métricas del mejor run: Accuracy = {best_run['metrics.Accuracy']:.4f}")

    # Cargar el modelo 
    model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # --- Evaluación del modelo cargado ---
    predictions = loaded_model.predict(X_unseen)
    
    final_accuracy = accuracy_score(y_unseen, predictions)
    final_precision = precision_score(y_unseen, predictions)
    final_recall = recall_score(y_unseen, predictions)
    final_f1 = f1_score(y_unseen, predictions)

    print("--- Métricas de Evaluación Final en Hold-Out Set ---")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1-Score: {final_f1:.4f}")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric("final_accuracy", final_accuracy)
        mlflow.log_metric("final_precision", final_precision)
        mlflow.log_metric("final_recall", final_recall)
        mlflow.log_metric("final_f1", final_f1)

    print("Métricas de evaluación final logueadas en el run de MLflow existente.")

if __name__ == "__main__":
    evaluate_model()