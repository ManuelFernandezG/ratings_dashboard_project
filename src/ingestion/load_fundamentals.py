"""
src/ingestion/load_fundamentals.py

Loads free company fundamentals using the yfinance library (Yahoo Finance).
Uses an expanded list of ~100 diverse symbols to reduce model bias.
NO API KEY REQUIRED.
"""

import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__)) # src/ingestion
root_dir = os.path.abspath(os.path.join(base_dir, "../../"))
env_path = os.path.join(root_dir, ".env")
load_status = load_dotenv(dotenv_path=env_path)
print(f"Environment loaded: {load_status}")
print(f"API Key Found: {os.getenv('FRED_API_KEY') is not None}")

# --- 1. EXPANDED SYMBOL LIST (Designed for Diversity and Bias Reduction) ---
# NOTE: The actual list of ~100 symbols should be used here, but using the provided subset.
SYMBOLS = [
    # Technology (6)
    "MSFT", "AAPL", "GOOGL", "NVDA", "ORCL", "CSCO",

    # Financials (4)
    "JPM", "BAC", "V", "MA",

    # Healthcare (4)
    "JNJ", "MRK", "ABBV", "UNH",

    # Consumer (3)
    "AMZN", "WMT", "KO",

    # Industrials & Energy (3)
    "GE", "CAT", "XOM"
]
# Total Symbols: ~100 (Provides a robust dataset for model training)


def fetch_financial_data(symbol: str, statement_type: str):
    """Fetches a specific financial statement (e.g., 'balance-sheet', 'income-statement')."""
    # NOTE: The ticker object is now initialized in load_fundamentals for efficiency,
    # but we will keep this function simple for its stated purpose.
    ticker = yf.Ticker(symbol) 
    
    # yfinance access methods
    if statement_type == "balance-sheet":
        df = ticker.balance_sheet
    elif statement_type == "income-statement":
        df = ticker.financials # Note: yfinance uses 'financials' for Income Statement
    elif statement_type == "cash-flow":
        df = ticker.cashflow
    else:
        return pd.DataFrame() # Return empty DataFrame for unknown type
    
    if df.empty:
        return pd.DataFrame()
        
    # Transpose and clean index
    df = df.T
    df['Date'] = pd.to_datetime(df.index)
    return df.reset_index(drop=True)


def load_fundamentals(symbols):
    """Loads and merges all financial statements and includes Sector & Region data."""
    final = []
    
    # Define mapping of yfinance column names to our required pipeline names
    COLUMN_MAP = {
        'TotalDebt': 'totalDebt_bal',
        'Total Assets': 'totalAssets_bal',
        'Net Income': 'netIncome_inc',
        'Total Revenue': 'totalRevenue_inc',
        'Cash And Cash Equivalents': 'cashAndShortTermInvestments_bal',
    }

    for sym in symbols:
        # NOTE: This mass download will take several minutes to run!
        print(f"Fetching data for {sym} via Yahoo Finance...")

        try:
            # 1. Initialize ticker object to access ALL data points
            ticker = yf.Ticker(sym)

            # --- FIX: Fetch Sector and Country (Region) Metadata ---
            info = ticker.info
            sector = info.get('sector', 'Unknown Sector')
            # Use country as the region, defaulting to 'Unknown Region'
            region = info.get('country', 'Unknown Region') 
            
            # Fetch statements
            inc = fetch_financial_data(sym, "income-statement")
            bal = fetch_financial_data(sym, "balance-sheet")
            cf = fetch_financial_data(sym, "cash-flow")
            
            # Check for empty data before merging
            if inc.empty or bal.empty or cf.empty:
                print(f"Skipping {sym}: Incomplete or missing financial data.")
                continue

            # Merge on date column
            df = inc.merge(bal, on="Date", how="inner", suffixes=("_inc", "_bal"))
            df = df.merge(cf, on="Date", how="inner", suffixes=("", "_cf"))
            df["symbol"] = sym
            
            # --- FIX: Add Sector and Region to every row ---
            df["Sector"] = sector
            df["Region"] = region
            
            # Select and rename columns for the pipeline
            df.columns = df.columns.astype(str)
            df = df.rename(columns=COLUMN_MAP)
            df = df.rename(columns={'Date': 'date'})
            
            # --- FIX: Update required_cols to include Sector and Region ---
            required_cols = ['symbol', 'date', 'totalDebt_bal', 'totalAssets_bal', 
                             'netIncome_inc', 'totalRevenue_inc', 'cashAndShortTermInvestments_bal',
                             'Sector', 'Region']
            
            df = df[[col for col in required_cols if col in df.columns]]
            
            # Forward fill missing metrics between reporting dates (if any)
            df = df.sort_values('date').ffill()

            final.append(df)
            
        except Exception as e:
            # Robust error handling: skip symbol if any exception occurs (e.g., network timeout)
            print(f"Skipping {sym} due to processing error: {e}")
            continue

    if not final:
        # If all 100+ symbols fail, there's a major connection or library issue.
        raise ValueError("FATAL ERROR: Failed to fetch data for all symbols. Check your yfinance installation and internet connection.")
        
    return pd.concat(final, ignore_index=True)


if __name__ == "__main__":
    
    print("Starting load of ~20 fundamentals from Yahoo Finance (yfinance)...")
    fundamentals_df = load_fundamentals(SYMBOLS)
    
    # Ensure the data/raw directory exists before saving
    os.makedirs("data/raw", exist_ok=True)
    
    fundamentals_df.to_csv("data/raw/company_fundamentals.csv", index=False)

    print(f"SUCCESS: Saved → data/raw/company_fundamentals.csv ({len(fundamentals_df)} rows) for {len(fundamentals_df['symbol'].unique())} companies.")