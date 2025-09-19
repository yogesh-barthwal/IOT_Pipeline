"""
===========================================================
IoT Predictive Maintenance Project — Simulated Data Source
===========================================================

⚠️ IMPORTANT:
This script generates FAKE / SYNTHETIC sensor data to mimic 
a smart factory environment. It does not connect to real 
machines or PLCs. All readings (temperature, vibration, 
pressure, current, etc.) are randomly simulated around 
machine-specific baselines for demonstration and testing 
purposes only.

The goal is to create a stream of realistic-looking data 
so we can practice:
- Data ingestion
- ETL with Dask
- ML model training/retraining
- Dashboard visualization

When deploying in a real plant, replace this simulation 
with actual IoT/PLC/SCADA data sources.

===========================================================
"""


"""
collect_data.py

Simulates a small fleet of machines with multiple sensors and appends
a CSV file per time-step into data/raw/. File-per-timestamp pattern
makes it easy to simulate incremental ingestion.

Config:
 - INTERVAL_SECONDS: how often to write a new batch (default 300s = 5min).
   For local dev set to e.g., 5 or 10 seconds.
 - BATCH_SIZE: number of rows per write (e.g., multiple machines)
"""

import os
import time
import uuid # to generate universally unique identifiers (UUIDs)-so we can tell records apart, even if two rows have the same values.
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw') 
#In Python, __file__ is a special variable that stores the path of the current file (the script being executed)
# os.path.dirname(__file__)strips off the filename and gives you the folder containing the file
# In paths, .. means “go up one directory”.

os.makedirs(RAW_DIR, exist_ok=True)

# --- Simulation parameters ---
INTERVAL_SECONDS = 10   # change to 300 for real 5-min cadence
NUM_MACHINES = 6
BATCH_SIZE = NUM_MACHINES
SEED = 42
np.random.seed(SEED)

# baseline sensor ranges per machine (simulate heterogeneity)- This code builds a list of dictionaries,
# one per machine, each dictionary holding the machine’s baseline sensor values.
# We want to simulate multiple machines in the factory.
# Each machine should have slightly different baseline sensor ranges, otherwise every machine would look identical and
# the simulation wouldn’t be realistic.

BASELINES = [
    {"temp": 60 + i*2, "vib": 0.5 + 0.05*i, "pressure": 1.0 + 0.02*i, "current": 5 + 0.5*i}
    for i in range(NUM_MACHINES)
]

def simulate_row(machine_id, baseline, usage_hours):
    # produce normal sensor readings with occasional drifting/noise
    # Each sensor reading is drawn from a normal distribution centered at the machine’s baseline value (loc).

    temp = np.random.normal(loc=baseline["temp"], scale=1.5)
    vib = np.abs(np.random.normal(loc=baseline["vib"], scale=0.15))
    pressure = np.random.normal(loc=baseline["pressure"], scale=0.05)
    current = np.random.normal(loc=baseline["current"], scale=0.4)

    # simulate usage increment- Increments machine usage by a random small amount each step (between 0.05 and 0.5 hours)
    usage_hours = usage_hours + np.random.uniform(0.05, 0.5)

    # simple failure generation: if temp, vib and current exceed thresholds -> failure
    failure_prob = 0.001

    # increase failure probability with drift- If any sensor drifts significantly beyond normal → failure probability jumps to 25%.
    # This is the “machine degradation” simulation.
    if temp > baseline["temp"] + 6 or vib > baseline["vib"] + 0.6 or current > baseline["current"] + 2:
        failure_prob = 0.25

    failure = np.random.rand() < failure_prob
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": machine_id,
        "temperature": round(float(temp), 3),
        "vibration": round(float(vib), 3),
        "pressure": round(float(pressure), 3),
        "current": round(float(current), 3),
        "usage_hours": round(float(usage_hours), 3),
        "failure": int(failure)
    }, usage_hours #A dictionary = one row of fake IoT sensor data.The updated usage_hours (to be used in the next row for continuity).

def main():
    # Initialize usage hours for each machine with a random starting value
    usage = {f"machine_{i}": float(np.random.uniform(100, 200)) for i in range(NUM_MACHINES)}
    
    # Map each machine to its baseline sensor values (temperature, vibration, etc.)
    baseline_map = {f"machine_{i}": BASELINES[i] for i in range(NUM_MACHINES)}

    print(f"Starting simulator: writing to {RAW_DIR}. Interval {INTERVAL_SECONDS}s") # Prints where the simulator will write .csv files and how often.
    try:
        while True:# Keeps generating new sensor data files at fixed intervals-Infinite loop (until stopped)
            rows = []
            # Generate one row of fake sensor data for each machine
            for i in range(NUM_MACHINES): #NUM_MACHINES is how many machines you’re simulating.
                mid = f"machine_{i}" # mid is just a string ID like "machine_0", "machine_1", etc.
                row, usage[mid] = simulate_row(mid, baseline_map[mid], usage[mid])
                rows.append(row)

            # Convert all machine readings into a DataFrame
            df = pd.DataFrame(rows)
            
            # Create a unique filename with timestamp + random suffix
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = os.path.join(RAW_DIR, f"sensor_{ts}_{uuid.uuid4().hex[:6]}.csv")
            
            # Save to CSV inside the raw data folder
            df.to_csv(filename, index=False)
            
            # Print log message with timestamp and file details
            print(f"[{datetime.now(timezone.utc).isoformat()}] Wrote {len(df)} rows -> {filename}")
            
            # Wait for the configured interval before generating the next batch
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        # Allow clean shutdown when user presses Ctrl+C
        print("Simulator stopped by user.")



if __name__ == "__main__":
    main()
