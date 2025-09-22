"""
retrain_loop.py

A simple retraining loop:
 - Periodically checks the raw data folder for new CSV files
 - If enough new files are present (>= MIN_RAW_FILES_TO_TRIGGER),
   it runs ETL and then retrains the model
 - Tracks which files have already been processed using a small JSON state file

This is a toy/dev setup. For production, consider:
 - A scheduler/flow system (Airflow, Prefect, cron)
 - Storing state in a database
 - Adding retries, logging, and alerting
"""

import os
import time
import json
from datetime import datetime

# --- Paths ---
ROOT = os.path.join(os.path.dirname(__file__), '..')          # project root
RAW_DIR = os.path.join(ROOT, 'data', 'raw')                   # folder containing raw CSVs
STATE_PATH = os.path.join(ROOT, 'retrain_state.json')         # JSON file to store processed file state

# --- Thresholds ---
MIN_RAW_FILES_TO_TRIGGER = 3   # trigger retraining if >= N new raw files found (3 for dev, higher for prod)
SLEEP_SECONDS = 15             # how often to poll (15s for dev, ~300s for 5min cadence in prod)

# --- Import ETL + training ---
from src.etl import run_etl
from src.train_model import train_and_save

# --- State management ---
def load_state():
    """Load retrain state (list of processed files) from JSON; create new if missing."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"processed_files": []}

def save_state(state):
    """Save retrain state to JSON file."""
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f)

# --- File discovery ---
def discover_new_files():
    """Return sorted list of all raw data files in the raw folder (CSV or Parquet)."""
    import glob
    files = sorted(
        glob.glob(os.path.join(RAW_DIR, '*.csv')) +
        glob.glob(os.path.join(RAW_DIR, '*.parquet'))
    )
    return files


# --- Main monitoring loop ---
def main_loop():
    # Load previously processed state
    state = load_state()
    last_processed = set(state.get("processed_files", []))

    print("Starting retrain loop. Monitoring raw files...")

    try:
        while True:
            # Discover all raw files
            files = discover_new_files()
            # Find which ones are new/unprocessed
            unprocessed = [f for f in files if f not in last_processed]

            # If enough new files -> run ETL + retrain
            if len(unprocessed) >= MIN_RAW_FILES_TO_TRIGGER:
                print(f"{len(unprocessed)} new files found -> running ETL + retrain")

                # Run ETL (convert raw -> processed parquet)
                success = run_etl(min_files=1, drop_raw_after=False)

                if success:
                    # Train model on updated processed dataset
                    model_path = train_and_save()

                    # Mark all files as processed
                    last_processed = set(files)
                    # Keep last 1000 processed filenames (avoid state.json growing too large)
                    state["processed_files"] = list(last_processed)[-1000:]
                    # Store retrain timestamp
                    state["last_trained_at"] = datetime.utcnow().isoformat()

                    # Save state back to disk
                    save_state(state)
                    print(f"Retrain completed and state saved. Model: {model_path}")

            else:
                # Not enough new files yet, wait and try again
                print(f"{len(unprocessed)} new files (need {MIN_RAW_FILES_TO_TRIGGER}). "
                      f"Sleeping {SLEEP_SECONDS}s.")

            # Sleep before checking again
            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("Retrain loop stopped by user.")

# --- Entry point ---
if __name__ == "__main__":
    main_loop()
