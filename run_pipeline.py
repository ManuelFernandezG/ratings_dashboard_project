# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:40:46 2025

@author: manny
"""

import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n--- Running: {script_path} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f" Error in {script_path}. Pipeline stopped.")
        sys.exit(1)
    print(f" Finished: {script_path}")

# 1. Ingestion (Yahoo Finance + FRED)
run_script("src/ingestion/load_fundamentals.py")

# 2. Preprocessing (Merging and Feature Building)
# Check if this file exists; it combines the fundamental data with FRED data
run_script("src/preprocessing/build_features.py")

# 3. Modeling (Training Logistic & XGBoost)
run_script("src/modeling/train_models.py")

# 4. Evaluation (Generating ROC, SHAP, and Metrics)
run_script("src/evaluation/evaluate.py")

print("\n🚀 FULL PIPELINE COMPLETE. You can now launch the dashboard.")