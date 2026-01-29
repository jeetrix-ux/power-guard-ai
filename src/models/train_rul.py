import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.feature_extractor import FeaturePipeline

def train_rul_model():
    print("Loading RUL Raw Data...")
    df_raw = pd.read_csv('data/rul_train_raw.csv')
    
    # Preprocessing for RUL
    # We generated "bursts" of 10 samples per cycle.
    # We need to aggregate these into a single row per cycle to map to RUL.
    # OR we can treat every sample as a valid predictor.
    # Let's aggregate to be robust (Mean/Std of the burst).
    
    print("Aggregating Cycles...")
    # Group by Life and Cycle
    grouped = df_raw.groupby(['life_id', 'cycle_id'])
    
    # Feature Engineering (Manual aggregation for RUL)
    # Ideally we use the Pipeline, but the Pipeline expects time-series.
    # Here we have bursts.
    # Let's calculate Mean Output Voltage, Mean Ripple, Mean Temp per cycle.
    
    X_df = grouped.agg({
        'V_out': ['mean', 'std'],
        'V_ripple_pkpk': ['mean'],
        'Temp_heatsink': ['mean'],
        # We can add more features here that correlate with degradation
    })
    
    # Flatten columns
    X_df.columns = ['_'.join(col).strip() for col in X_df.columns.values]
    
    # Target RUL
    y = grouped['RUL'].first()
    
    # Split Train/Val
    # Split by Life ID to prevent leakage
    lives = df_raw['life_id'].unique()
    train_lives = lives[:int(0.8 * len(lives))]
    
    # Create mask using the MultiIndex (life_id is level 0)
    train_mask = X_df.index.get_level_values('life_id').isin(train_lives)
    
    X_train = X_df[train_mask]
    y_train = y[train_mask]
    
    X_test = X_df[~train_mask]
    y_test = y[~train_mask]
    
    print(f"Training RUL Regressor on {len(X_train)} cycles...")
    
    # Model
    regr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    regr.fit(X_train, y_train)
    
    # Evaluate
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRUL Model Performance:")
    print(f" - MAE: {mae:.2f} cycles")
    print(f" - R2 Score: {r2:.3f}")
    
    # Save
    joblib.dump(regr, 'src/models/rul_regressor.pkl')
    print("Saved model to src/models/rul_regressor.pkl")

if __name__ == "__main__":
    train_rul_model()
