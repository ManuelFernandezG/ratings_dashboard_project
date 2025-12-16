# -*- coding: utf-8 -*-
"""
src/evaluation/evaluate.py

Loads saved models (Logistic Regression and XGBoost) from models/ and test data.
Calculates core credit risk metrics: AUC, ROC, KS statistic, and confusion matrix.
Includes robust calculation of SHAP values for interpretability data.
"""
# -*- coding: utf-8 -*-
"""
src/evaluation/evaluate.py

Loads saved models (Logistic Regression and XGBoost) from models/ and test data.
Calculates core credit risk metrics: AUC, ROC, KS statistic, and confusion matrix.
Includes robust calculation of SHAP values for interpretability data using a 
future-proof Generic Explainer to bypass XGBoost string-to-float errors.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap

# --- 1. Setup Paths ---
# Ensures files are saved in the project-level 'models' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
DATA_PATH = os.path.join(MODEL_DIR, "test_data_for_evaluation.csv")

def run_evaluation():
    # Load test data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Test data not found at {DATA_PATH}. Run training script first.")
        return
    
    test_df = pd.read_csv(DATA_PATH)
    X_test = test_df.drop(columns=['TARGET_HIGH_RISK'])
    y_test = test_df['TARGET_HIGH_RISK']

    performance_summary = {}

    # Models to evaluate
    model_files = {
        "Logistic_Regression": "log_reg_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }

    for name, filename in model_files.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            print(f"Skipping {name}: File not found.")
            continue

        model = joblib.load(model_path)
        
        # Get predictions
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        # 1. Calculate ROC & AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Save ROC data for the dashboard
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        roc_df.to_csv(os.path.join(MODEL_DIR, f"{name.lower()}_roc_data.csv"), index=False)

        # 2. Calculate KS Statistic
        ks_stat = max(tpr - fpr)
        
        # 3. Confusion Matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

        performance_summary[name] = {
            "AUC": float(roc_auc),
            "KS_Statistic": float(ks_stat),
            "True_Positives": int(tp),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn)
        }

        # --- 4. SHAP Logic (Specific to XGBoost) ---
        if name == "XGBoost":
            print("Calculating SHAP values for XGBoost (Kernel/Generic Mode)...")
            
            # This bypasses the buggy XGBoost tree loader causing 'could not convert string to float'
            # We wrap the model in a lambda to predict probabilities
            model_func = lambda x: model.predict_proba(x)[:, 1]
            
            # Use a small background sample to speed up calculation
            bg_data = X_test.sample(min(100, len(X_test)), random_state=42)
            
            # Use Generic Explainer which is robust to XGBoost/SHAP version mismatches
            explainer = shap.Explainer(model_func, bg_data)
            
            # Sample first 500 rows for SHAP (calculating for 36k+ rows is too slow)
            sample_size = min(500, len(X_test))
            shap_values = explainer(X_test.head(sample_size))
            
            # Save SHAP values (the actual impact scores) to CSV
            shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
            
            # Handle expected_value (base value)
            if hasattr(explainer, 'expected_value'):
                ev = explainer.expected_value
                shap_df['expected_value'] = ev[0] if isinstance(ev, (list, np.ndarray)) else ev
            else:
                shap_df['expected_value'] = 0.5 
            
            shap_df.to_csv(os.path.join(MODEL_DIR, "xgboost_shap_values.csv"), index=False)
            print(" SUCCESS: SHAP values saved using Generic Explainer.")

    # Save overall performance summary
    summary_path = os.path.join(MODEL_DIR, "model_performance_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(performance_summary, f, indent=4)
    
    print(f"Evaluation complete. Results saved to {MODEL_DIR}")

if __name__ == "__main__":
    run_evaluation()