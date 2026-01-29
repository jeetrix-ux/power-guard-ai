import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.buck_converter import BuckConverterSimulator

def generate_rul_data(num_lives=50):
    print(f"Generating Run-to-Failure Data ({num_lives} Simulated Lives)...")
    
    all_runs = []
    
    # Failure Threshold: ESR > 4.0x Nominal
    FAILURE_THRESHOLD = 4.0
    
    for life_id in range(num_lives):
        sim = BuckConverterSimulator()
        
        # Random initial healthy state
        # Some perfectly new, some slightly used
        current_esr_mult = np.random.uniform(1.0, 1.2)
        sim.set_degradation(esr_mult=current_esr_mult)
        
        # Simulation parameters
        # We simulate "Virtual Hours" by stepping degradation faster than physics
        # We assume 1 simulation step = 1 hour of operational life for degradation purposes
        # But physics is still computed at dt
        
        life_data = []
        time_step = 0
        max_steps = 1000 # Safety break
        
        # Usage Profile
        v_in = 24.0
        duty = 0.5
        load = 10.0
        
        # Degradation Rate (Variable per unit to prevent overfitting)
        deg_rate = np.random.uniform(0.005, 0.01) 
        
        while current_esr_mult < FAILURE_THRESHOLD and time_step < max_steps:
            # Run short physics burst to measuring sensors
            # We take 0.1s of data to compute features
            # Note: We need to use the exact feature extraction logic later
            # For efficiency here, we will capture raw sensor readings 
            # and let the feature pipeline handle it later.
            # But the feature pipeline expects a time-series window.
            
            # To make this efficient:
            # We record ONE row per "Degradation Step"
            # This row represents the "Average behavior" at this health state.
            
            # Run 10 steps of physics to get noise/ripple
            burst_data = []
            for _ in range(10):
                d = sim.step(v_in, duty, load, 0.01)
                burst_data.append(d)
                
            # Compute a simple aggregated row for training the RUL
            # Ideally we use the SAME pipeline, but that needs a DataFrame.
            # Let's create a mini-dataframe
            df_burst = pd.DataFrame(burst_data)
            
            # We'll just save the raw burst. 
            # Actually, to train, we need to generate features.
            # Let's save the RAW burst data with a "Life_ID" and "Cycle_ID".
            # This allows the robust FeaturePipeline to process it correctly.
            
            df_burst['life_id'] = life_id
            df_burst['cycle_id'] = time_step
            
            # Current Health State (Hidden from Model, used for RUL calc)
            # RUL = (Time of Failure - Current Time)
            # We don't know Time of Failure yet, so we store and backfill later.
            
            life_data.append(df_burst)
            
            # Degrade
            current_esr_mult += deg_rate * np.random.uniform(0.8, 1.2) # Stochastic degradation
            sim.set_degradation(esr_mult=current_esr_mult)
            time_step += 1
            
        # Compile Life
        df_life = pd.concat(life_data, ignore_index=True)
        
        # Calculate RUL
        # The last cycle_id is the failure point
        max_cycle = df_life['cycle_id'].max()
        df_life['RUL'] = max_cycle - df_life['cycle_id']
        
        all_runs.append(df_life)
        
        if (life_id + 1) % 10 == 0:
            print(f" - Simulating Life {life_id + 1}/{num_lives} (Duration: {max_cycle} cycles)")

    full_df = pd.concat(all_runs, ignore_index=True)
    
    # Save raw RUL data
    full_df.to_csv('data/rul_train_raw.csv', index=False)
    print(f"Saved {len(full_df)} samples to data/rul_train_raw.csv")

if __name__ == "__main__":
    generate_rul_data()
