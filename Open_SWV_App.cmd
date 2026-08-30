@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv310\Scripts\python.exe" set "PYTHON_EXE=.venv310\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "env\Scripts\python.exe" set "PYTHON_EXE=env\Scripts\python.exe"

if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=py"
)
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3.10+ from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed for:
    echo   %PYTHON_EXE%
    echo.
    choice /C YN /M "Install the app requirements now"
    if errorlevel 2 exit /b 1
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting Electrochemistry Analysis App...
echo.
"%PYTHON_EXE%" -m streamlit run app.py

endlocal
