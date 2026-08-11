@echo off
setlocal enabledelayedexpansion
title A-SASS Speech Therapy Launcher
cls

echo ============================================
echo   A-SASS Speech Therapy System
echo ============================================
echo.

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "APP_FILE=%SCRIPT_DIR%app.py"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

:: --- 1. Check Python 3.10 / 3.11 ---
echo [+] Checking Python installation...

set "PYTHON_EXE="

py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 set "PYTHON_EXE=py -3.11"

if not defined PYTHON_EXE (
    py -3.10 --version >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_EXE=py -3.10"
)

if not defined PYTHON_EXE (
    python --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
        echo !PY_VER! | findstr /R "^3\.10\." >nul && set "PYTHON_EXE=python"
        echo !PY_VER! | findstr /R "^3\.11\." >nul && set "PYTHON_EXE=python"
    )
)

if defined PYTHON_EXE goto :PYTHON_FOUND

echo [!] Python 3.10 or 3.11 not found on system.
echo [+] Downloading Python 3.11.9 (64-bit)...

set "PY_INSTALLER=%TEMP%\python-3.11.9-amd64.exe"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%PY_INSTALLER%' } catch { exit 1 }"

if not exist "%PY_INSTALLER%" (
    echo [ERROR] Failed to download Python installer. Check internet connection.
    goto :PAUSE_EXIT
)

echo [+] Installing Python 3.11.9 silently...
start /wait "" "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1
del /f /q "%PY_INSTALLER%" >nul 2>&1

set "PATH=%LocalAppData%\Programs\Python\Python311;%LocalAppData%\Programs\Python\Python311\Scripts;%PATH%"

py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 set "PYTHON_EXE=py -3.11"

if not defined PYTHON_EXE (
    python --version >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python installation completed but could not be detected in PATH.
    goto :PAUSE_EXIT
)

:PYTHON_FOUND
echo [OK] Using Python: %PYTHON_EXE%

:: --- 2. Check App Files ---
if not exist "%APP_FILE%" (
    echo [ERROR] app.py not found in directory: %SCRIPT_DIR%
    goto :PAUSE_EXIT
)
echo [OK] app.py found.

:: --- 3. Clean Running Processes ---
echo [+] Cleaning up existing Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 >nul

:: --- 4. Virtual Environment Setup ---
if exist "%VENV_DIR%" (
    if not exist "%VENV_DIR%\Scripts\activate.bat" (
        echo [!] Broken virtual environment detected. Cleaning up...
        rmdir /s /q "%VENV_DIR%" 2>nul
    )
)

if not exist "%VENV_DIR%" (
    echo [+] Creating fresh virtual environment...
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :PAUSE_EXIT
    )
    echo [OK] Virtual environment created.
)

echo [+] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    goto :PAUSE_EXIT
)
echo [OK] Virtual environment activated.

:: --- 5. Install Dependencies ---
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip setuptools wheel -q >nul 2>&1

echo [+] Installing required packages...
echo [i] First run downloads AI/Whisper dependencies (~5-15 mins)...
python -m pip install -r "%REQ_FILE%"
if !errorlevel! neq 0 (
    echo [ERROR] Package installation failed. Please check internet connection.
    goto :PAUSE_EXIT
)
echo [OK] All packages installed.

:: --- 6. Launch App ---
echo.
echo [OK] Launching Streamlit interface...
echo [i] Opening browser at http://localhost:8501
echo ============================================
echo.

start /b cmd /c "timeout /t 6 >nul && start http://localhost:8501"
streamlit run "%APP_FILE%" --server.headless=false

:PAUSE_EXIT
echo.
echo ============================================
echo Execution paused. Press any key to exit script.
echo ============================================
pause
