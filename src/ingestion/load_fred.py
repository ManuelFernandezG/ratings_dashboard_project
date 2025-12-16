# -*- coding: utf-8 -*-
"""
src/ingestion/load_fred.py
Load macro, credit-spread, and CDS data from FRED using absolute paths.
"""

import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv

# --- 1. ROBUST PATH SETUP ---
# Finds the root directory (ratings_dashboard_project) relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src/ingestion
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

# Force load the .env from the root folder
load_dotenv(dotenv_path=ENV_PATH)
FRED_API_KEY = os.getenv("FRED_API_KEY") 

# Safety check
if FRED_API_KEY is None:
    raise ValueError(f"FRED_API_KEY not found. Ensure it's in: {ENV_PATH}")

def load_fred_series(series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Load a single FRED time series and forward-fill NaNs."""
    fred = Fred(api_key=FRED_API_KEY)
    try:
        data = fred.get_series(series_id, observation_start=start_date)
        df = pd.DataFrame({"date": data.index, series_id: data.values})
        df["date"] = pd.to_datetime(df["date"])
        # MACRO DATA TIP: macro series often have missing weekend/holiday values
        df[series_id] = df[series_id].ffill() 
        return df
    except Exception as e:
        print(f"Error loading {series_id}: {e}")
        return pd.DataFrame(columns=["date", series_id])

def load_credit_data():
    """Loads credit spread proxies and macro variables, merging on date."""
    series_list = {
        "HIGH_YIELD_OAS": "BAMLH0A0HYM2", 
        "IG_OAS": "BAMLC0A0CM",        
        "TED_SPREAD": "TEDRATE",       
        "VIX": "VIXCLS",               
        "FED_FUNDS": "FEDFUNDS",       
        "US_10Y_RATE": "DGS10"         
    }

    dfs = []
    for label, sid in series_list.items():
        print(f"Fetching {label} ({sid})...")
        df = load_fred_series(sid)
        df = df.rename(columns={sid: label})
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Outer merge to align all dates (FRED series frequencies often vary)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")
        
    # Sort by date and forward-fill gaps created by merging different frequencies
    merged = merged.sort_values("date").ffill()
    return merged

if __name__ == "__main__":
    print("--- Starting FRED Macro & Credit Data Load ---")
    fred_df = load_credit_data()

    if not fred_df.empty:
        # Define the absolute output path
        output_dir = os.path.join(ROOT_DIR, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "fred_credit_data.csv")
        fred_df.to_csv(output_file, index=False)
        print(f" SUCCESS: Saved → {output_file} ({len(fred_df)} rows)")
    else:
        print(" FAILED: No data was loaded from FRED.")