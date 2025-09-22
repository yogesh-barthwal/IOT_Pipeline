"""
etl.py

A Dask-based ETL that:
 - reads all CSVs in data/raw/
 - concatenates to a single Dask DataFrame
 - basic cleaning
 - feature engineering: per-machine rolling mean/std (window over recent timestamps)
 - writes a consolidated Parquet at data/processed/processed.parquet

You can run it manually (python src/etl.py) or call from retrain_loop.
"""

import os
from datetime import datetime, timedelta
import glob
import shutil  # for optional file/directory operations

import dask.dataframe as dd
import pandas as pd

# Define project directories
ROOT = os.path.join(os.path.dirname(__file__), '..')  # project root
RAW_DIR = os.path.join(ROOT, 'data', 'raw')           # folder with raw CSVs
PROC_DIR = os.path.join(ROOT, 'data', 'processed')    # folder for processed data
os.makedirs(PROC_DIR, exist_ok=True)                 # create processed folder if missing

PARQUET_PATH = os.path.join(PROC_DIR, 'processed.parquet')  # final Parquet file

def run_etl(min_files=None, drop_raw_after=False):
    """
    Run the ETL process:
    - min_files: minimum number of raw CSVs required to run
    - drop_raw_after: whether to delete raw files after processing
    """
    
    # list all CSV files in RAW_DIR
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv'))) # Collects all CSV file paths from RAW_DIR, matches '*.csv' using glob,
                                                                  # joins the path safely with os.path, sorts them alphabetically for consistency,
                                                                  # and stores the result as a list


                                                                  # glob is a module in Python’s standard library.
                                                                  # # It’s used to search for files and directories that match a specific pattern.

    
    # if not enough files, skip ETL
    if min_files and len(csv_files) < min_files:
        print(f"Found {len(csv_files)} raw files; waiting for at least {min_files}. Exiting ETL.")
        return False

    if not csv_files:
        print("No raw CSV files to process.")
        return False

    print(f"ETL: reading {len(csv_files)} files")
    
    # --- Extract: read all CSVs efficiently with Dask ---
    ddf = dd.read_csv(
        csv_files,
        assume_missing=True,
        dtype={
            "machine_id": "object", 
            "temperature": "float64",
            "vibration": "float64",
            "pressure": "float64",
            "current": "float64",
            "usage_hours": "float64",
            "failure": "int64"
        }
    )

    # --- Transform ---
    # convert timestamp column to datetime (UTC)
    ddf['timestamp'] = dd.to_datetime(ddf['timestamp'], utc=True)
    
    # sort each partition by machine_id and timestamp
    ddf = ddf.map_partitions(lambda df: df.sort_values(['machine_id', 'timestamp'])) # Sorts rows by machine_id and timestamp within each partition (local sort, not global)
    
    # drop rows with missing critical sensor values
    ddf = ddf.dropna(subset=['temperature', 'vibration', 'pressure', 'current', 'usage_hours'])

    # --- Feature engineering ---
    # Rolling statistics per machine (mean, std over last 5 rows)
    def add_rolling_features(pdf): # pdf- Pandas Data Frame
        pdf = pdf.sort_values('timestamp')
        window = 5  # last 5 readings
        pdf['temp_mean_5'] = pdf['temperature'].rolling(window=window, min_periods=1).mean() 
        
        # rolling(window=5) looks at the current row + 4 previous rows.
        # min_periods=1 means it will still compute even if fewer than 5 rows exist (useful at the beginning).
        # mean() → moving average (smooths noise).
        # std() → moving variability (detects anomalies).
        # .fillna(0) → replaces NaN values from std (like when only 1 row exists)

        pdf['temp_std_5'] = pdf['temperature'].rolling(window=window, min_periods=1).std().fillna(0)
        pdf['vib_mean_5'] = pdf['vibration'].rolling(window=window, min_periods=1).mean()
        pdf['vib_std_5'] = pdf['vibration'].rolling(window=window, min_periods=1).std().fillna(0)
        pdf['current_mean_5'] = pdf['current'].rolling(window=window, min_periods=1).mean()
        pdf['current_std_5'] = pdf['current'].rolling(window=window, min_periods=1).std().fillna(0)
        
        # time in hours since first record per machine
        pdf['ts_hours'] = (pdf['timestamp'] - pdf['timestamp'].min()).dt.total_seconds() / 3600.0 # difference between each timestamp and the very first timestamp.
        return pdf

    # apply rolling feature computation per machine using Dask groupby-apply
    #ddf.groupby('machine_id')
    # Groups the big Dask DataFrame (ddf) by machine_id.
    # This ensures each machine’s data is handled separately.

    grouped = ddf.groupby('machine_id').apply(
        add_rolling_features,
        meta=ddf._meta.assign(                                                      # Dask is lazy (doesn’t execute immediately).
            # To build the computation graph, it needs to know the structure (columns + dtypes) of the DataFrame
            # that add_rolling_features will return.That’s why we provide meta.
            temp_mean_5='f8', temp_std_5='f8', # 'f8' → shorthand for float64 dtype.
            vib_mean_5='f8', vib_std_5='f8',
            current_mean_5='f8', current_std_5='f8',
            ts_hours='f8'
        )
    )

    # --- Load ---
    print("Computing and writing Parquet...")
    grouped = grouped.persist()  # compute Dask graph- .persist() tells Dask to start computing the graph now and store the result in memory (RAM) across its workers.
    grouped.to_parquet(PARQUET_PATH, engine='pyarrow', overwrite=True)
    print(f"Processed data written to {PARQUET_PATH}")

    # optionally delete raw CSV files after ETL
    if drop_raw_after:
        for f in csv_files:
            os.remove(f)
        print(f"Removed {len(csv_files)} raw files after ETL.")

    return True

# Run ETL when script is executed directly
if __name__ == "__main__":
    run_etl()
