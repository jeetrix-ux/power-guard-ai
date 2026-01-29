import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.feature_extractor import FeaturePipeline

def train_models():
    print("Loading Datasets...")
    df_healthy = pd.read_csv('data/healthy_train.csv')
    df_faulty = pd.read_csv('data/faulty_train.csv')
    
    # 1. Feature Engineering
    print("1. Fitting Feature Pipeline...")
    pipeline = FeaturePipeline(window_size=10) # 100ms window at 100Hz
    
    # Fit on HEALTHY data only (Standard Scaler needs to know what 'Normal' looks like)
    pipeline.fit(df_healthy)
    
    X_healthy = pipeline.transform(df_healthy)
    X_faulty = pipeline.transform(df_faulty)
    
    # Save Pipeline
    joblib.dump(pipeline, 'src/models/feature_pipeline.pkl')
    
    # 2. Anomaly Detection (Unsupervised)
    print("2. Training Anomaly Detector (Isolation Forest)...")
    # Contamination very low because training set is pure healthy
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X_healthy)
    
    # Verify on Faulty data (Should be -1)
    y_pred_faulty = iso_forest.predict(X_faulty)
    detection_rate = (y_pred_faulty == -1).mean()
    print(f"   -> Detection Rate on Faulty Data: {detection_rate:.2%}")
    
    joblib.dump(iso_forest, 'src/models/anomaly_detector.pkl')
    
    # 3. Fault Classification (Supervised)
    print("3. Training Fault Classifier (XGBoost)...")
    y_faulty = df_faulty.loc[X_faulty.index, 'fault_label']
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_faulty)
    
    # Train
    clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    clf.fit(X_faulty, y_encoded)
    
    # Evaluate
    y_pred = clf.predict(X_faulty)
    print("\nClassification Report (Training Data):")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))
    
    # Save Model and Encoder
    joblib.dump(clf, 'src/models/fault_classifier.pkl')
    joblib.dump(le, 'src/models/label_encoder.pkl')
    
    print("\nAll models trained and saved to src/models/")

if __name__ == "__main__":
    train_models()
