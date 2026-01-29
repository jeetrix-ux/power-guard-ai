import pandas as pd
import numpy as np
import joblib
import os
import sys

class HealthMonitor:
    def __init__(self, model_dir='src/models'):
        self.model_dir = model_dir
        self.pipeline = joblib.load(os.path.join(model_dir, 'feature_pipeline.pkl'))
        self.anomaly_detector = joblib.load(os.path.join(model_dir, 'anomaly_detector.pkl'))
        self.classifier = joblib.load(os.path.join(model_dir, 'fault_classifier.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        self.rul_regressor = joblib.load(os.path.join(model_dir, 'rul_regressor.pkl'))
        
        # Buffer for rolling features
        # We need at least 'window_size' samples. The pipeline uses window=10.
        self.window_size = 10 
        self.data_buffer = [] 
        
    def process_stream(self, data_point: dict):
        """
        Process a single dictionary of sensor readings.
        Returns: None (buffering) or Result Dictionary
        """
        self.data_buffer.append(data_point)
        
        # Maintain buffer size slightly larger than window to be safe if diffing
        # Actually pipeline creates NaNs for first 10, so we need >10 samples to get 1 valid row?
        # Rolling(10) requires 10 samples to produce the 10th value.
        # Diff() requires 1 previous.
        # So we need at least 11 samples roughly.
        
        KEEP_SIZE = 20
        if len(self.data_buffer) > KEEP_SIZE:
            self.data_buffer.pop(0)
            
        if len(self.data_buffer) < self.window_size + 1:
            return {"status": "Buffering", "color": "gray"}
            
        # Convert buffer to DataFrame
        df_buffer = pd.DataFrame(self.data_buffer)
        
        # Transform
        # We only care about the LAST row of the transformed data
        try:
            # We transform the whole buffer, but only use the tail
            # This is inefficient for high-frequency but fine for demo (100Hz)
            X_all = self.pipeline.transform(df_buffer)
            
            if len(X_all) == 0:
                 return {"status": "Buffering", "color": "gray"}
                 
            # Take the last sample
            X_latest = X_all.iloc[[-1]] 
            
            # 1. Anomaly Detection
            # Isolation Forest: 1 = Normal, -1 = Anomaly
            is_normal = self.anomaly_detector.predict(X_latest)[0] == 1
            anomaly_score = self.anomaly_detector.decision_function(X_latest)[0]
            
            if is_normal:
                # Calculate RUL even if healthy (Prognostics)
                rul_feats = pd.DataFrame([{
                    'V_out_mean': df_buffer['V_out'].mean(),
                    'V_out_std': df_buffer['V_out'].std(),
                    'V_ripple_pkpk_mean': df_buffer['V_ripple_pkpk'].mean(),
                    'Temp_heatsink_mean': df_buffer['Temp_heatsink'].mean()
                }])
                predicted_rul = self.rul_regressor.predict(rul_feats)[0]
                
                return {
                    "status": "Healthy",
                    "color": "green",
                    "anomaly_score": round(anomaly_score, 3),
                    "fault_type": "None",
                    "RUL_prediction": round(predicted_rul, 1)
                }
            else:
                # 2. Fault Classification
                fault_idx = self.classifier.predict(X_latest)[0]
                fault_name = self.label_encoder.inverse_transform([fault_idx])[0]
                proba = np.max(self.classifier.predict_proba(X_latest))
                
                
                # 3. RUL Estimator (Run always or only if healthy?)
                # We developed RUL model on "Aggregated Cycle Data".
                # We need to compute [V_out_mean, V_out_std, V_ripple_pkpk_mean, Temp_heatsink_mean]
                # from the buffer.
                
                # Compute features matching training time
                rul_feats = pd.DataFrame([{
                    'V_out_mean': df_buffer['V_out'].mean(),
                    'V_out_std': df_buffer['V_out'].std(),
                    'V_ripple_pkpk_mean': df_buffer['V_ripple_pkpk'].mean(),
                    'Temp_heatsink_mean': df_buffer['Temp_heatsink'].mean()
                }])
                
                predicted_rul = self.rul_regressor.predict(rul_feats)[0]
                
                return {
                    "status": "Fault Detected",
                    "color": "red",
                    "anomaly_score": round(anomaly_score, 3),
                    "fault_type": fault_name,
                    "confidence": round(proba, 2),
                    "RUL_prediction": round(predicted_rul, 1)
                }
                
        except Exception as e:
            # Usually happens during startup of pipeline (NaNs)
            return {"status": "Processing...", "color": "gray", "debug": str(e)}

