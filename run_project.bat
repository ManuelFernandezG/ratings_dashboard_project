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