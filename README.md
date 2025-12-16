# 📊 Credit Risk Dashboard & ML Pipeline

This project is an end-to-end Machine Learning solution for corporate credit risk assessment. It automates the ingestion of financial fundamentals and macroeconomic data, trains predictive models (XGBoost & Logistic Regression), and visualizes model performance and interpretability (SHAP) through an interactive Dash dashboard.

## 🚀 One-Click Execution

If you have already installed the requirements (see below), you can run the entire pipeline—from data cleaning to dashboard launch—using a single command.

### **Windows**

1.  Navigate to the project folder.
2.  Double-click **`run_project.bat`**. (This script automatically searches for Miniconda/Anaconda on your system).

### **macOS / Linux**

1.  Open Terminal and navigate to the project folder.
2.  Run: `bash run_project.sh`

-----

## 🛠️ Manual Installation & Setup

### 1\. Environment Setup

Create a virtual environment named `credit-risk-env` to manage dependencies:

```bash
# Create environment
conda create -n credit-risk-env python=3.10 -y

# Activate environment
conda activate credit-risk-env

# Install dependencies
pip install -r requirements.txt
```

### 2\. API Configuration (Critical)

This project requires a free FRED API key to fetch macroeconomic data.

1.  Get your key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).
2.  Create a file named exactly **`.env`** in the root directory (ensure it is not `.env.txt`).
3.  Add your key without quotes or spaces:
    ```text
    FRED_API_KEY=your_32_character_key_here
    ```

-----

## 📂 Project Structure

| Folder | Description |
| :--- | :--- |
| `src/ingestion/` | Pulls raw financial data via `yfinance` and macro indicators via `FRED` API. |
| `src/preprocessing/` | Joins annual fundamentals with daily macro data; engineers credit ratios. |
| `src/modeling/` | Training scripts for Logistic Regression and XGBoost. |
| `src/evaluation/` | Metric calculation (AUC/KS) and SHAP value generation. |
| `dashboards/` | Multi-page Dash application code. |
| `data/` | Local storage for raw/processed datasets (Created automatically). |
| `models/` | Serialized `.pkl` models and evaluation JSONs (Created automatically). |

-----

## 🏗️ Portable Script for Windows (`run_project.bat`)

This script is designed for portability. It dynamically searches for Conda in standard user and system paths.

```batch
@echo off
SETLOCAL
TITLE Credit Risk Pipeline - Universal Launcher

:: Find the project folder
SET "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: DYNAMIC CONDA SEARCH
IF EXIST "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    SET "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\activate.bat"
) ELSE IF EXIST "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    SET "CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat"
) ELSE IF EXIST "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    SET "CONDA_PATH=C:\ProgramData\miniconda3\Scripts\activate.bat"
) ELSE (
    echo [ERROR] Could not find Miniconda or Anaconda. Please install and try again.
    pause
    exit /b
)

:: ACTIVATE ENVIRONMENT
call "%CONDA_PATH%" credit-risk-env

:: Ensure folder structure exists
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models

echo --- CLEANING OLD DATA ---
if exist "data\raw\*.csv" del /q "data\raw\*.csv"
if exist "data\processed\*.csv" del /q "data\processed\*.csv"
if exist "models\*.csv" del /q "models\*.csv"
if exist "models\*.json" del /q "models\*.json"
if exist "models\*.pkl" del /q "models\*.pkl"

echo --- RUNNING PIPELINE ---
python src/ingestion/load_fundamentals.py
python src/ingestion/load_fred.py
python src/preprocessing/build_features.py
python src/modeling/train_models.py
python src/evaluation/evaluate.py

echo --- LAUNCHING DASHBOARD ---
start http://127.0.0.1:8050/
cd dashboards
python app.py
pause
```

-----

## 🏗️ Script for macOS/Linux (`run_project.sh`)

```bash
#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

# Ensure folder structure exists
mkdir -p data/raw data/processed models

source $(conda info --base)/etc/profile.d/conda.sh
conda activate credit-risk-env

echo "--- CLEANING OLD DATA ---"
rm -f data/raw/*.csv data/processed/*.csv models/*.csv models/*.json models/*.pkl

echo "--- RUNNING PIPELINE ---"
python src/ingestion/load_fundamentals.py
python src/ingestion/load_fred.py
python src/preprocessing/build_features.py
python src/modeling/train_models.py
python src/evaluation/evaluate.py

echo "--- LAUNCHING DASHBOARD ---"
cd dashboards
python app.py
```

-----

## 🧠 Technical Methodology

The pipeline transforms raw financial statements into predictive risk signals through four key stages:

### 1\. Data Fusion

We combine **Annual Fundamentals** (static company health) with **Daily Macro Data** (dynamic market stress). To bridge the frequency gap, annual ratios are forward-filled to create a daily time series.

### 2\. Synthetic Labeling (The Target)

The model predicts a **90-day forward-looking credit event**. A "High Risk" event is defined when the BofA High Yield Option-Adjusted Spread (OAS) increases by more than **500 basis points** within the subsequent 3-month window.

### 3\. Machine Learning Strategy

  * **XGBoost**: Used to capture non-linear relationships.
  * **Logistic Regression**: Used for a stable, linear baseline.

### 4\. Interpretability (SHAP)

We utilize **SHAP (Shapley Additive Explanations)** to break down each company's risk score into specific "contributions" from features like Debt-to-Equity or the current TED Spread.