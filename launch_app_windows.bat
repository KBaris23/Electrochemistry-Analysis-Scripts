@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "SYSTEM_PYTHON="
set "VENV_DIR=.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    rem Prefer a 64-bit interpreter. NumPy, pandas, and SciPy no longer provide
    rem reliable current Windows packages for 32-bit Python.
    for %%I in (
        "%USERPROFILE%\anaconda3\envs\ea-bo-analysis\python.exe"
        "%USERPROFILE%\anaconda3\envs\centris311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\miniconda3\python.exe"
    ) do (
        if not defined SYSTEM_PYTHON if exist "%%~I" (
            "%%~I" -c "import struct,sys; sys.exit(struct.calcsize('P') == 4)" >nul 2>&1
            if not errorlevel 1 set "SYSTEM_PYTHON=%%~I"
        )
    )

    rem Also check interpreters available on PATH, resolving their full paths.
    for /f "delims=" %%I in ('where py 2^>nul') do (
        if not defined SYSTEM_PYTHON (
            "%%I" -c "import struct,sys; sys.exit(struct.calcsize('P') == 4)" >nul 2>&1
            if not errorlevel 1 set "SYSTEM_PYTHON=%%I"
        )
    )
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if not defined SYSTEM_PYTHON (
            "%%I" -c "import struct,sys; sys.exit(struct.calcsize('P') == 4)" >nul 2>&1
            if not errorlevel 1 set "SYSTEM_PYTHON=%%I"
        )
    )

    if not defined SYSTEM_PYTHON (
        echo A 64-bit Python installation was not found.
        echo Install 64-bit Python 3.10 or newer and try again.
        pause
        exit /b 1
    )

    echo Creating a Python virtual environment using:
    echo   !SYSTEM_PYTHON!
    "!SYSTEM_PYTHON!" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON=%VENV_DIR%\Scripts\python.exe"

echo Installing required packages...
"%PYTHON%" -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo Failed to install the required packages.
    pause
    exit /b 1
)

echo Starting the Electrochemistry Analysis app...
echo Your browser should open at http://localhost:8501
"%PYTHON%" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo The app stopped with an error.
    pause
)

endlocal
