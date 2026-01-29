import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.buck_converter import BuckConverterSimulator
from src.utils.visualizer import plot_telemetry

def profile(t):
    # Dynamic Load Profile
    v_in = 24.0
    duty = 0.5 # Target 12V
    
    # Load Steps
    if t < 5.0:
        load = 10.0 # 1.2A
    else:
        load = 5.0  # 2.4A
        
    return v_in, duty, load

def main():
    print("Initializing Simulation...")
    sim = BuckConverterSimulator()
    
    # Phase 1: Healthy
    print("Running Healthy Phase (0-10s)...")
    df_healthy = sim.run_profile(10.0, 100.0, profile)
    
    # Phase 2: Degraded Capacitor
    print("Injecting Fault: Capacitor Aging (ESR x 3)...")
    sim.set_degradation(esr_mult=3.0)
    df_faulty = sim.run_profile(5.0, 100.0, profile)
    
    # Combine
    full_df = pd.concat([df_healthy, df_faulty], ignore_index=True)
    
    print(f"Simulation Complete. Generated {len(full_df)} samples.")
    print(full_df.tail())
    
    plot_telemetry(full_df, "Simulation Validation: Healthy -> Load Step -> Cap Degradation")
    print("Plot saved to simulation_check.png")

if __name__ == "__main__":
    main()
