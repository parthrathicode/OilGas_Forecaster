@echo off
REM One-click launcher for Windows: creates the virtual environment on first run,
REM installs dependencies, then starts the dashboard.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    echo Installing dependencies, this can take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

".venv\Scripts\python.exe" -m streamlit run src\dashboard.py %*
