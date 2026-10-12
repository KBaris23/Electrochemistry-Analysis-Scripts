@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Keep the managed environment outside the repository. This avoids checking
rem installed wheels into Git and prevents an unrelated local .venv from being
rem selected accidentally.
set "APP_SUPPORT_DIR=%LOCALAPPDATA%\Electrochemistry Analysis Scripts"
set "VENV_DIR=%APP_SUPPORT_DIR%\venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

if exist "%PYTHON%" (
    "%PYTHON%" -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 and (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Replacing an incompatible app environment:
        echo   %VENV_DIR%
        rmdir /s /q "%VENV_DIR%"
        if exist "%VENV_DIR%" (
            echo Could not replace the incompatible app environment.
            pause
            exit /b 1
        )
    )
)

if not exist "%PYTHON%" (
    set "SYSTEM_PYTHON="
    set "SYSTEM_PYTHON_ARGS="

    rem Prefer stable 64-bit CPython installations with binary scientific wheels.
    for %%I in (
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%USERPROFILE%\anaconda3\envs\ea-bo-analysis\python.exe"
        "%USERPROFILE%\anaconda3\envs\centris311\python.exe"
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\miniconda3\python.exe"
    ) do (
        if not defined SYSTEM_PYTHON if exist "%%~I" (
            "%%~I" -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 and (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set "SYSTEM_PYTHON=%%~I"
        )
    )

    rem Support installations registered only through the Windows py launcher.
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        for %%V in (3.11 3.10 3.12 3.13) do (
            if not defined SYSTEM_PYTHON (
                py -%%V-64 -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 and (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
                if !errorlevel! equ 0 (
                    set "SYSTEM_PYTHON=py"
                    set "SYSTEM_PYTHON_ARGS=-%%V-64"
                )
            )
        )
    )

    rem Finally inspect ordinary PATH entries, but still enforce 64-bit Python.
    for /f "delims=" %%I in ('where python 2^>nul') do (
        if not defined SYSTEM_PYTHON (
            "%%I" -c "import struct,sys; sys.exit(0 if struct.calcsize('P') == 8 and (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set "SYSTEM_PYTHON=%%I"
        )
    )

    if not defined SYSTEM_PYTHON (
        echo A compatible 64-bit Python installation was not found.
        echo Install 64-bit Python 3.10, 3.11, 3.12, or 3.13 and try again.
        pause
        exit /b 1
    )

    echo Creating the managed app environment using:
    echo   !SYSTEM_PYTHON! !SYSTEM_PYTHON_ARGS!
    if not exist "%APP_SUPPORT_DIR%" mkdir "%APP_SUPPORT_DIR%"
    "!SYSTEM_PYTHON!" !SYSTEM_PYTHON_ARGS! -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create the managed app environment.
        pause
        exit /b 1
    )
)

echo Checking app requirements for:
echo   %PYTHON%
"%PYTHON%" -m pip install --disable-pip-version-check --upgrade "pip>=24,<27"
if errorlevel 1 (
    echo Failed to update pip.
    pause
    exit /b 1
)
"%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

"%PYTHON%" -c "import struct,sys,numpy,pandas,scipy,streamlit; assert struct.calcsize('P') == 8; print('Python', sys.version.split()[0], '64-bit; dependencies ready.')"
if errorlevel 1 (
    echo Installed dependency verification failed.
    pause
    exit /b 1
)

if /I "%SWV_LAUNCHER_SETUP_ONLY%"=="1" (
    echo Setup-only verification completed successfully.
    exit /b 0
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
