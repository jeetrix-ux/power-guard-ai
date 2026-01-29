import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Industrial-grade feature extractor for Power Electronics.
    
    Generates:
    - Time-domain statistics (Rolling Mean, Std)
    - Physics-based features (Power, Efficiency Proxy)
    - Rate-of-change features (Derivatives)
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fit(self, X, y=None):
        # Generate features on X to learn scaling
        X_feat = self._generate_features(X)
        self.feature_names = X_feat.columns.tolist()
        self.scaler.fit(X_feat)
        return self
        
    def transform(self, X):
        X_feat = self._generate_features(X)
        return pd.DataFrame(self.scaler.transform(X_feat), columns=self.feature_names, index=X_feat.index)
        
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core physics-aware feature engineering.
        """
        # copy to avoid SettingWithCopy warnings
        df = df.copy()
        
        # 1. Physics Features (Instantaneous interaction)
        # Power calculation
        df['P_in'] = df['V_in'] * df['V_out'] / 24.0 # Estimate input current roughly or just use V_in
        df['P_out'] = df['V_out'] * df['I_load']
        # Thermal Rise vs Power (Efficiency proxy)
        df['Temp_Rise_per_Watt'] = (df['Temp_heatsink'] - 25.0) / (df['P_out'] + 1e-6)
        
        # 2. Rolling Statistics (Trend analysis)
        # We assume the dataframe is time-ordered
        cols_to_roll = ['V_out', 'I_load', 'V_ripple_pkpk', 'Temp_heatsink']
        
        for col in cols_to_roll:
            # Simple rolling mean/std
            df[f'{col}_mean_{self.window_size}'] = df[col].rolling(window=self.window_size).mean()
            df[f'{col}_std_{self.window_size}'] = df[col].rolling(window=self.window_size).std()
            
            # Rate of Change (first difference)
            df[f'{col}_roc'] = df[col].diff()
            
        # 3. Drop NaNs created by rolling/diff
        # In production, we would use a buffer. Here we fill or drop.
        # For training, dropping 5 rows is fine.
        df_clean = df.dropna()
        
        # Select purely numeric feature columns (exclude timestamps, raw labels if present)
        # We kept physics features + rolling stats
        drop_cols = ['timestamp', 'V_in', 'ESR_internal_truth', 'R_ds_internal_truth', 'fault_label', 'P_in']
        # Note: We keep V_out, I_load etc as raw features too often, 
        # but let's stick to the engineered ones + raw sensor inputs.
        
        final_features = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns], errors='ignore')
        
        return final_features

