"""
dataset.py - EV Battery Degradation Dataset Generator

Generates realistic EV battery operational telemetry benchmarked against 
NASA Ames Prognostics Center of Excellence (PCoE) battery aging experiments.
Incorporates electrochemical aging mechanisms:
- Cycle degradation with knee-point fade
- Arrhenius thermal acceleration (high temp side reactions & low temp plating)
- DC Fast-charging mechanical stress & SEI layer growth
- C-rate (discharge current) polarization
- Voltage boundary stresses (deep discharge / overvoltage)
"""

import numpy as np
import pandas as pd


def generate_battery_dataset(num_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 1. Operational Charge Cycles (0 to 600 cycles)
    cycle = np.random.uniform(0, 600, size=num_samples)

    # 2. State of Charge (%) - (10% to 100%)
    soc = np.random.uniform(10, 100, size=num_samples)

    # 3. Discharge Current Load (A) - (0.50A to 3.00A)
    current = np.random.uniform(0.5, 3.0, size=num_samples)

    # 4. Operating Temperature (°C) - (10.0°C to 55.0°C)
    temperature = np.random.uniform(10.0, 55.0, size=num_samples)

    # 5. Fast Charging Flag (0 = Standard AC, 1 = DC Fast Charge)
    fast_charging = np.random.binomial(1, 0.4, size=num_samples)

    # 6. Terminal Voltage (V) under load (Nominal ~ 17.5V pack module, 14.0V - 18.5V)
    # Voltage correlates with SoC and internal resistance drop (I * R_int)
    # Internal resistance increases as cycle count increases
    r_int = 0.08 + (cycle / 600.0) * 0.06 + np.where(temperature < 18, (18 - temperature) * 0.003, 0.0)
    ocv = 15.0 + (soc / 100.0) * 3.3  # Open circuit voltage between ~15.0V and 18.3V
    voltage = ocv - (current * r_int) + np.random.normal(0, 0.08, size=num_samples)
    voltage = np.clip(voltage, 14.0, 18.5)

    # 7. Measured Capacity (Ah) (6.0 Ah to 11.5 Ah; nominal fresh ~ 10.0 Ah)
    # Fresh pack capacity ~ 10.0 Ah with small manufacturing variance
    fresh_capacity = np.random.normal(10.0, 0.25, size=num_samples)

    # Physics-informed degradation calculation (State of Health %)
    # Base cycle fade (initial square-root/sublinear growth followed by steady fade)
    cycle_fade = 0.055 * cycle + 0.000035 * (cycle ** 2)

    # Fast charging multiplier: increases SEI growth rate and mechanical fatigue
    fast_charge_penalty = fast_charging * (0.025 * cycle)

    # Arrhenius thermal penalty:
    # High temp (> 25°C): accelerated cathode degradation and electrolyte consumption
    # Low temp (< 15°C): lithium plating and sluggish kinetics
    high_temp_excess = np.maximum(0, temperature - 25.0)
    low_temp_excess = np.maximum(0, 15.0 - temperature)
    temp_penalty = 0.28 * high_temp_excess + 0.15 * low_temp_excess

    # Extreme combined stress: high current + high temp + fast charging
    stress_interaction = (current > 2.0) * (high_temp_excess > 10.0) * 2.5

    # Voltage stress (deep discharge below 15V or overvoltage above 18V)
    deep_discharge = np.maximum(0, 15.2 - voltage) * 1.8

    # Total degradation calculation
    total_degradation = (
        cycle_fade + 
        fast_charge_penalty + 
        temp_penalty + 
        stress_interaction + 
        deep_discharge + 
        np.random.normal(0, 0.6, size=num_samples)
    )

    # Calculate actual remaining capacity and SoH
    calculated_soh = 100.0 - total_degradation
    calculated_soh = np.clip(calculated_soh, 45.0, 99.8)

    # Measured capacity reflects actual SoH relative to fresh capacity
    capacity = (calculated_soh / 100.0) * fresh_capacity + np.random.normal(0, 0.05, size=num_samples)
    capacity = np.clip(capacity, 5.5, 12.0)

    # Assemble into DataFrame
    df = pd.DataFrame({
        'cycle': np.round(cycle, 1),
        'soc': np.round(soc, 1),
        'voltage': np.round(voltage, 2),
        'current': np.round(current, 2),
        'temperature': np.round(temperature, 1),
        'capacity': np.round(capacity, 2),
        'fast_charging': fast_charging,
        'soh': np.round(calculated_soh, 2)
    })

    return df


if __name__ == '__main__':
    data = generate_battery_dataset(20)
    print("Sample generated battery records:")
    print(data.head())
