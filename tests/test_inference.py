import sys
import os
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.inference import HealthMonitor

def test_inference_logic():
    print("Initializing Health Monitor...")
    monitor = HealthMonitor()
    
    print("Loading Test Stream...")
    df = pd.read_csv('data/test_stream.csv')
    
    print(f"Streaming {len(df)} samples...")
    results = []
    
    start_time = time.time()
    for _, row in df.iterrows():
        # Convert row to dict
        data_point = row.to_dict()
        res = monitor.process_stream(data_point)
        if res:
            results.append(res['status'])
            
    elapsed = time.time() - start_time
    print(f"Processed in {elapsed:.2f}s ({len(df)/elapsed:.0f} Hz)")
    
    # Check simple stats
    healthy_count = results.count('Healthy')
    fault_count = results.count('Fault Detected')
    
    print(f"Healthy: {healthy_count}")
    print(f"Faults: {fault_count}")
    
    if fault_count > 0:
        print("PASS: System detected faults in the stream.")
    else:
        print("FAIL: No faults detected (did you train the models?)")

if __name__ == "__main__":
    test_inference_logic()
