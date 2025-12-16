
# -*- coding: utf-8 -*-
"""
src/preprocessing/build_features.py
Cleans, transforms, and engineers features using absolute paths.
"""

import pandas as pd
import numpy as np
import os
from fredapi import Fred
from dotenv import load_dotenv

# --- 1. ROBUST PATH CONFIGURATION ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, "../../"))
load_dotenv(os.path.join(root_dir, ".env"))

# Define Absolute Paths for Data
RAW_FUNDAMENTALS = os.path.join(root_dir, "data", "raw", "company_fundamentals.csv")
RAW_FRED = os.path.join(root_dir, "data", "raw", "fred_credit_data.csv")
PROCESSED_DIR = os.path.join(root_dir, "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features_and_target.csv")

FUNDAMENTAL_COLS = [
    "symbol", "date", "totalDebt_bal", "totalAssets_bal", 
    "netIncome_inc", "totalRevenue_inc", "cashAndShortTermInvestments_bal",
]

# --- 2. HELPER FUNCTIONS ---

def calculate_credit_ratios(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating financial ratios...")
    numeric_cols = [col for col in df.columns if col not in ['symbol', 'date']]
    for col in numeric_cols: 
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Leverage, Profitability, and Liquidity
    df["LEVERAGE"] = df["totalDebt_bal"] / df["totalAssets_bal"] if 'totalDebt_bal' in df.columns else np.nan
    df["PROFIT_MARGIN"] = df["netIncome_inc"] / df["totalRevenue_inc"] if 'netIncome_inc' in df.columns else np.nan
    df["LIQUIDITY_RATIO"] = df["cashAndShortTermInvestments_bal"] / df["totalAssets_bal"] if 'totalAssets_bal' in df.columns else np.nan

    raw_cols_to_drop = ['totalDebt_bal', 'totalAssets_bal', 'netIncome_inc', 'totalRevenue_inc', 'cashAndShortTermInvestments_bal']
    return df.drop(columns=[c for c in raw_cols_to_drop if c in df.columns], errors='ignore')

def construct_synthetic_label(df: pd.DataFrame) -> pd.DataFrame:
    if "HIGH_YIELD_OAS" not in df.columns:
        df["TARGET_HIGH_RISK"] = np.nan
        return df

    # Signal: 1 if Spread > 500bps
    df["HIGH_RISK_ALERT"] = np.where(pd.to_numeric(df["HIGH_YIELD_OAS"], errors="coerce") > 5.0, 1, 0)
    # Target: Signal shifted back 90 days (forward-looking prediction)
    df["TARGET_HIGH_RISK"] = df["HIGH_RISK_ALERT"].shift(-90)
    return df.drop(columns=["HIGH_RISK_ALERT"], errors='ignore')

# --- 3. MAIN EXECUTION ---

def build_features():
    print("--- Starting Feature Engineering ---")
    
    # Load Raw Data using Absolute Paths
    try:
        fred_df = pd.read_csv(RAW_FRED)
        fundamentals_df = pd.read_csv(RAW_FUNDAMENTALS)
    except FileNotFoundError as e:
        print(f" Error: {e}. Run ingestion scripts first.")
        return None

    # Date Conversion
    fred_df["date"] = pd.to_datetime(fred_df["date"])
    fundamentals_df["date"] = pd.to_datetime(fundamentals_df["date"])

    # 1. Ratios and Data Replication (Daily Frequency)
    available_cols = [col for col in FUNDAMENTAL_COLS if col in fundamentals_df.columns]
    fundamentals_ratios = calculate_credit_ratios(fundamentals_df[available_cols].copy())
    fundamentals_ratios = fundamentals_ratios.set_index('date').sort_index()

    full_date_range = pd.date_range(start=fundamentals_ratios.index.min(), end=fred_df['date'].max(), freq='D')
    
    dense_fundamentals = []
    for symbol, group_df in fundamentals_ratios.groupby('symbol'):
        group_df = group_df.reindex(full_date_range).ffill()
        group_df['symbol'] = symbol
        dense_fundamentals.append(group_df.dropna(subset=['symbol'])) 

    final_df = pd.concat(dense_fundamentals).reset_index().rename(columns={'index': 'date'})
    
    # 2. Merge with Macro Data
    merged_df = pd.merge(final_df, fred_df, on="date", how="left")
    macro_cols = [c for c in fred_df.columns if c != 'date']
    merged_df[macro_cols] = merged_df[macro_cols].ffill() 
    
    # 3. Label and Cleaning
    final_df = merged_df.groupby('symbol').apply(construct_synthetic_label).reset_index(level=0, drop=True)
    final_df['TARGET_HIGH_RISK'] = final_df.groupby('symbol')['TARGET_HIGH_RISK'].ffill()
    final_df.dropna(subset=["TARGET_HIGH_RISK"], inplace=True)
    
    final_df = final_df.fillna(final_df.median(numeric_only=True)).fillna(0)

    # 4. Save Processed Data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f" SUCCESS: Saved processed features → {OUTPUT_FILE} ({len(final_df)} rows)")
    return final_df

if __name__ == "__main__":
    build_features()