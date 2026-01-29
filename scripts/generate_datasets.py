import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.buck_converter import BuckConverterSimulator

def random_profile(t):
    # Stochastic use patterns
    # V_in varies slowly 23V - 25V
    v_in = 24.0 + np.sin(t * 0.1) * 1.0 + np.random.normal(0, 0.1)
    
    # Duty cycle roughly fixed for 12V out, but control loop would adjust it.
    # We simulate closed loop by adjusting D to target 12V approx
    target_v = 12.0
    # D = V_out / V_in
    duty = target_v / v_in
    
    # Load jumps random
    # Base load 5 Ohm
    # Add random steps
    load = 5.0
    if int(t) % 10 < 5: 
        load = 2.0 # Heavy load
    else:
        load = 10.0 # Light load
        
    return v_in, duty, load

def generate_healthy():
    print("Generating Healthy Training Data...")
    sim = BuckConverterSimulator()
    # 60 seconds, 100 Hz = 6000 samples
    df = sim.run_profile(60.0, 100.0, random_profile)
    df.to_csv('data/healthy_train.csv', index=False)
    print(f"Saved {len(df)} samples to data/healthy_train.csv")

def generate_faults():
    print("Generating Fault Class Training Data...")
    dfs = []
    
    # Fault 1: Capacitor Aging
    print("- Simulating Capacitor Aging...")
    sim = BuckConverterSimulator()
    sim.set_degradation(esr_mult=5.0, c_mult=0.7)
    df = sim.run_profile(20.0, 100.0, random_profile)
    dfs.append(df)
    
    # Fault 2: MOSFET Degradation
    print("- Simulating MOSFET Wear...")
    sim = BuckConverterSimulator()
    sim.set_degradation(rds_mult=3.0)
    df = sim.run_profile(20.0, 100.0, random_profile)
    dfs.append(df)
    
    # Fault 3: Thermal Runaway
    print("- Simulating Thermal Overload...")
    sim = BuckConverterSimulator() # Default
    # Force temp high manually or via blocked cooling path
    # We can hack the sim state or parameters
    # Let's just run it with high resistance and low load to heat it up fast? 
    # Or just inject starting temp
    sim.temperature = 90.0
    df = sim.run_profile(20.0, 100.0, random_profile)
    dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv('data/faulty_train.csv', index=False)
    print(f"Saved {len(full_df)} samples to data/faulty_train.csv")

def generate_test_stream():
    print("Generating Test Stream (drift scenario)...")
    sim = BuckConverterSimulator()
    
    # 0-30s: Healthy
    df1 = sim.run_profile(30.0, 100.0, random_profile)
    
    # 30-60s: Mild Aging (ESR 2x) (Drift)
    sim.set_degradation(esr_mult=2.0)
    df2 = sim.run_profile(30.0, 100.0, random_profile)
    
    # 60-90s: Severe Aging (ESR 5x)
    sim.set_degradation(esr_mult=5.0)
    df3 = sim.run_profile(30.0, 100.0, random_profile)
    
    full_df = pd.concat([df1, df2, df3], ignore_index=True)
    full_df.to_csv('data/test_stream.csv', index=False)
    print(f"Saved {len(full_df)} samples to data/test_stream.csv")

if __name__ == "__main__":
    generate_healthy()
    generate_faults()
    generate_test_stream()
