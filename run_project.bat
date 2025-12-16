@echo off
SETLOCAL
TITLE Credit Risk Dashboard - Portable Pipeline

:: 1. DYNAMIC PATHING: Find the folder where this .bat file lives
SET "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: 2. ACTIVATE ENVIRONMENT 
:: We use 'call' so the script continues after the environment activates
:: Note: This assumes 'conda' is in the user's System PATH.
call conda activate credit-risk-env

:: Fallback check if conda isn't in PATH (standard Miniconda location)
if %errorlevel% neq 0 (
    echo [INFO] Standard conda command failed. Trying default Miniconda path...
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" credit-risk-env
)

echo ==================================================
echo   PROJECT ROOT: %PROJECT_ROOT%
echo   CLEANING OLD DATA AND MODELS
echo ==================================================
if exist "data\raw\*.csv" del /q "data\raw\*.csv"
if exist "data\processed\*.csv" del /q "data\processed\*.csv"
if exist "models\*.csv" del /q "models\*.csv"
if exist "models\*.json" del /q "models\*.json"
if exist "models\*.pkl" del /q "models\*.pkl"

echo ==================================================
echo   PHASE 1: DATA INGESTION
echo ==================================================
python src/ingestion/load_fundamentals.py
python src/ingestion/load_fred.py

echo ==================================================
echo   PHASE 2: FEATURE ENGINEERING
echo ==================================================
python src/preprocessing/build_features.py

echo ==================================================
echo   PHASE 3: MODEL TRAINING
echo ==================================================
python src/modeling/train_models.py

echo ==================================================
echo   PHASE 4: EVALUATION AND SHAP
echo ==================================================
python src/evaluation/evaluate.py

echo ==================================================
echo   PHASE 5: LAUNCHING DASHBOARD
echo ==================================================
cd dashboards
python app.py

pause