# src/data_prep.py
import pandas as pd
import os

def prepare_data():
    df = pd.read_csv('data/raw/telco_churn.csv')
    
    df_clean = df.copy()
    
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    df_clean.to_csv(os.path.join(output_dir, 'telco_churn_processed.csv'), index=False)

if __name__ == "__main__":
    prepare_data()
