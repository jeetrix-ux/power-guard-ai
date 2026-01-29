import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ComponentParams:
    """Nominal parameters for the Buck Converter."""
    L_inductance: float = 100e-6    # 100 uH
    C_capacitance: float = 470e-6   # 470 uF
    R_L_dcr: float = 0.05           # Inductor DC resistance
    R_C_esr_nominal: float = 0.05   # Capacitor Nominal ESR
    R_ds_on_nominal: float = 0.02   # MOSFET On-resistance
    f_sw: float = 100e3             # 100 kHz switching frequency
    V_diode: float = 0.7            # Diode forward voltage drop
    R_thermal: float = 5.0          # Thermal resistance (degC/W)
    C_thermal: float = 10.0         # Thermal heat capacity (J/degC)
    T_ambient: float = 25.0         # Ambient temperature

class BuckConverterSimulator:
    """
    Behavioral Simulator for a Buck Converter.
    
    Physics Model:
    - Calculates steady-state ripple and DC operating points based on switching parameters.
    - Updates thermal state based on power losses (conduction + switching).
    - Supports parameter degradation simulation (Health Physics).
    """
    
    def __init__(self, params: ComponentParams = ComponentParams()):
        self.params = params
        
        # State Variables
        self.current_time = 0.0
        self.temperature = params.T_ambient
        self.current_V_out = 0.0
        
        # Degradation Multipliers (1.0 = Healthy)
        self.esr_multiplier = 1.0
        self.c_multiplier = 1.0
        self.rds_multiplier = 1.0
        
        # Random State
        self.rng = np.random.default_rng(seed=42)

    def set_degradation(self, esr_mult: float = 1.0, c_mult: float = 1.0, rds_mult: float = 1.0):
        """Inject faults by modifying component physics."""
        self.esr_multiplier = esr_mult
        self.c_multiplier = c_mult
        self.rds_multiplier = rds_mult

    def step(self, V_in: float, duty_cycle: float, load_resistance: float, dt: float) -> Dict[str, float]:
        """
        Advance the simulation by dt seconds.
        
        Returns:
            Dictionary of sensor readings.
        """
        p = self.params
        
        # 1. Apply Degradation
        eff_esr = p.R_C_esr_nominal * self.esr_multiplier
        eff_c = p.C_capacitance * self.c_multiplier
        eff_rds = p.R_ds_on_nominal * self.rds_multiplier
        
        # 2. Electrical Physics (Steady State Approximation for this timestep)
        # We assume the dynamics settle faster than the logging dt for the DC point,
        # but we calculate AC characteristics explicitly.
        
        # Ideal Output Voltage
        # V_out = D * V_in - (Losses)
        # Simple approximation including diode and resistive drops
        # V_out_approx = V_in * D
        
        # Load Current (Iterative for P_loss interaction roughly)
        # Assuming V_out stabilizes close to target for current calc
        v_target = V_in * duty_cycle
        i_load = v_target / (load_resistance + 1e-6)
        
        # Accurate DC Transfer function with losses:
        # V_out = (V_in * D - V_diode*(1-D)) - I_load * (D*R_ds + (1-D)*R_diode + R_L)
        # Simplified for robust simulation:
        drop_resistive = i_load * (duty_cycle * eff_rds + p.R_L_dcr)
        v_dc = (V_in * duty_cycle) - drop_resistive
        
        # Inductor Ripple Current (Peak-to-Peak)
        # dI = (V_in - V_out) * D / (L * f)
        ripple_current_pkpk = max(0, (V_in - v_dc) * duty_cycle / (p.L_inductance * p.f_sw))
        
        # Output Voltage Ripple (Peak-to-Peak)
        # dV = dI * ESR + dI / (8 * C * f)
        ripple_voltage_esr = ripple_current_pkpk * eff_esr
        ripple_voltage_cap = ripple_current_pkpk / (8 * eff_c * p.f_sw)
        ripple_voltage_pkpk = ripple_voltage_esr + ripple_voltage_cap
        
        # 3. Thermal Physics (Dynamic)
        # Power Loss Calculation
        # P_cond_fet = D * I^2 * Rds
        p_cond_fet = duty_cycle * (i_load**2) * eff_rds
        # P_cond_ind = I^2 * R_L
        p_cond_ind = (i_load**2) * p.R_L_dcr
        # Switching Loss (Simplified linear approx)
        # P_sw = 0.5 * V_in * I_load * (t_rise + t_fall) * f
        # Assuming fixed transition time fraction for simplicity factor k_sw
        k_sw_loss_factor = 20e-9 # 20ns combined rise/fall roughly
        p_sw = 0.5 * V_in * i_load * k_sw_loss_factor * p.f_sw
        
        total_power_loss = p_cond_fet + p_cond_ind + p_sw
        
        # Thermal ODE Update: dT = (P_loss - P_dissipated)/C_th * dt
        # P_dissipated = (T - T_amb) / R_th
        p_dissipated = (self.temperature - p.T_ambient) / p.R_thermal
        delta_temp = ((total_power_loss - p_dissipated) / p.C_thermal) * dt
        
        self.temperature += delta_temp
        self.current_time += dt
        
        # 4. Sensor Synthesis (Adding Reality Gap/Noise)
        # Add sensor noise related to thermal (Johnson limits not strictly necessary but gaussian is good)
        
        noise_v = self.rng.normal(0, 0.005) # 5mV noise
        noise_i = self.rng.normal(0, 0.01)  # 10mA noise
        noise_t = self.rng.normal(0, 0.1)   # 0.1C measure noise
        
        # Provide "Instantaneous" snapshot reading (DC + random phase of ripple)
        # For ML, we often care about the statistical properties of the ripple, 
        # so we will return the "Measured Ripple" explicitly as a feature the DAQ system calculates,
        # OR we can simulate high-freq sampling. 
        # To respect the user's "Industry Grade", we will simulate an Advanced DAQ 
        # that reports both Mean values and High-Freq AC metrics.
        
        return {
            "timestamp": self.current_time,
            "V_in": V_in,
            "V_out": v_dc + noise_v,
            "I_load": i_load + noise_i,
            "V_ripple_pkpk": ripple_voltage_pkpk * self.rng.normal(1.0, 0.05), # 5% measurement error on ripple
            "Temp_heatsink": self.temperature + noise_t,
            "ESR_internal_truth": eff_esr,       # For labelling/debug (not fed to ML)
            "R_ds_internal_truth": eff_rds       # For labelling/debug
        }

    def run_profile(self, duration_sec: float, sampling_rate_hz: float, 
                   profile_func=None) -> pd.DataFrame:
        """
        Run simulation for a duration.
        profile_func(t) -> (V_in, Duty, Load_R)
        """
        dt = 1.0 / sampling_rate_hz
        steps = int(duration_sec / dt)
        
        records = []
        
        # Defaults
        v_in = 24.0
        duty = 0.5
        load = 10.0
        
        for _ in range(steps):
            if profile_func:
                v_in, duty, load = profile_func(self.current_time)
            
            readings = self.step(v_in, duty, load, dt)
            readings['fault_label'] = self._determine_label()
            records.append(readings)
            
        return pd.DataFrame(records)

    def _determine_label(self) -> str:
        """Simple ground-truth labeler based on degradation state."""
        if self.esr_multiplier > 1.5 or self.c_multiplier < 0.8:
            return "Capacitor_Degradation"
        if self.rds_multiplier > 1.3:
            return "MOSFET_Degradation" # Could lead to thermal runaway
        if self.temperature > 85.0: # Arbitrary thermal limit
            return "Overheat"
        return "Healthy"
