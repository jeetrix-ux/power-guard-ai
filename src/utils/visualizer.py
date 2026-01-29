import matplotlib.pyplot as plt
import pandas as pd

def plot_telemetry(df: pd.DataFrame, title: str = "System Telemetry"):
    """Plot key sensor data from the simulation dataframe."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # 1. Voltage
    axes[0].plot(df['timestamp'], df['V_out'], label='V_out', color='blue')
    axes[0].plot(df['timestamp'], df['V_in'], label='V_in', color='green', alpha=0.5, linestyle='--')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Voltage Regulation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Current
    axes[1].plot(df['timestamp'], df['I_load'], label='I_load', color='orange')
    axes[1].set_ylabel('Current (A)')
    axes[1].set_title('Load Current')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Ripple (Health Indicator)
    axes[2].plot(df['timestamp'], df['V_ripple_pkpk'] * 1000, label='V_ripple (mV)', color='purple')
    axes[2].set_ylabel('Ripple (mV)')
    axes[2].set_title('Output Voltage Ripple (Condition Indicator)')
    axes[2].grid(True, alpha=0.3)

    # 4. Temperature
    axes[3].plot(df['timestamp'], df['Temp_heatsink'], label='Temp (C)', color='red')
    axes[3].axhline(y=85, color='black', linestyle='--', label='Warning Threshold')
    axes[3].set_ylabel('Temp (°C)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Thermal Status')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('simulation_check.png')
    plt.close()
