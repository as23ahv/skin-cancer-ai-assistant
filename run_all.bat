@echo off
cd /d "%~dp0"

echo ===============================
echo Final Year Project Runner
echo ===============================

REM --- Use THIS project's venv python directly (no activate needed)
set VENV_PY=%CD%\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
  echo ERROR: Could not find venv python at:
  echo %VENV_PY%
  echo Make sure your venv folder is named ".venv"
  pause
  exit /b 1
)

echo Using: %VENV_PY%
echo.

echo Exporting metrics...
"%VENV_PY%" model\export_metrics.py

echo.
echo Starting Streamlit app...
"%VENV_PY%" -m streamlit run chatbot\app.py

pause
