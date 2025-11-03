# src/train.py
from pycaret.classification import *
import yaml
import pandas as pd
import os
import mlflow
from dotenv import load_dotenv

def train_model():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    if params.get('track_to_dagshub', False):
        load_dotenv()
        mlflow.set_tracking_uri(params['dagshub_tracking_uri'])
    else:
        if 'MLFLOW_TRACKING_URI' in os.environ:
            del os.environ['MLFLOW_TRACKING_URI']
        if 'MLFLOW_TRACKING_USERNAME' in os.environ:
            del os.environ['MLFLOW_TRACKING_USERNAME']
        if 'MLFLOW_TRACKING_PASSWORD' in os.environ:
            del os.environ['MLFLOW_TRACKING_PASSWORD']
        
    # Cargar datos procesados
    df = pd.read_csv(params['data_read_csv'])
    
    # Setup PyCaret
    exp = ClassificationExperiment()
    exp.setup(
        data=df, 
        target='churn',
        train_size=params['train_size'],
        session_id=params['seed'],
        log_experiment=True, # Activar logging a MLflow
        experiment_name="telco-churn-prediction"
    )
    
    best_models = exp.compare_models(
        include=params['models_to_compare'],
        sort=params['metric'],
        n_select=3
    )
    
    best_model = exp.tune_model(best_models[0])
    
    final_model = exp.finalize_model(best_model)
    

if __name__ == "__main__":
    train_model()
