@echo off
chcp 65001 >nul
title A-SASS Speech Therapy Launcher
cls

echo ============================================
echo   A-SASS Speech Therapy System
echo ============================================
echo.

:: --- 1. Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install from python.org and check "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] Python 3 found.

:: --- 2. Paths (quoted for spaces) ---
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "APP_FILE=%SCRIPT_DIR%app.py"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"

:: --- 3. Check app.py ---
if not exist "%APP_FILE%" (
    echo [ERROR] app.py not found in: %SCRIPT_DIR%
    pause
    exit /b 1
)
echo [OK] app.py found.

:: --- 4. KILL any python processes locking the venv ---
echo [+] Cleaning up old processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 >nul

:: --- 5. FORCE DELETE old venv if it exists ---
if exist "%VENV_DIR%" (
    echo [!] Removing old venv folder...
    rmdir /s /q "%VENV_DIR%" 2>nul
    :: If rmdir failed (permission locked), try renaming
    if exist "%VENV_DIR%" (
        echo [!] Folder locked. Renaming instead...
        ren "%VENV_DIR%" "venv_old_%random%" 2>nul
        if exist "%VENV_DIR%" (
            echo [ERROR] Cannot remove or rename venv.
            echo Fix: Close all Python windows, then retry.
            echo Or: Move this entire folder to Desktop and retry.
            pause
            exit /b 1
        )
    )
)

:: --- 6. Create fresh venv ---
echo [+] Creating fresh virtual environment...
python -m venv "%VENV_DIR%" --upgrade-deps
if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    echo Trying fallback: installing without venv...
    goto :NO_VENV_FALLBACK
)
echo [OK] venv created.

:: --- 7. Activate venv ---
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv.
    goto :NO_VENV_FALLBACK
)
echo [OK] venv activated.

:: --- 8. Ensure pip works ---
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing...
)

:: --- 9. Install requirements ---
echo [+] Installing packages (first run: 5-15 minutes)...
echo [i] This will download Whisper models (~1-3 GB)
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Package install failed. Check internet connection.
    pause
    exit /b 1
)
echo [OK] All packages installed.
goto :LAUNCH

:: --- FALLBACK: No venv, install to user site ---
:NO_VENV_FALLBACK
echo.
echo [!] Falling back to system Python (no virtual environment)...
python -m pip install --user -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Installation failed completely.
    echo Please run this file as Administrator (Right-click ^> Run as administrator)
    pause
    exit /b 1
)
echo [OK] Packages installed to user profile.

:: --- 10. Launch Streamlit ---
:LAUNCH
echo.
echo [✓] Starting Streamlit...
echo [i] Browser will open at http://localhost:8501
echo ============================================
echo.

:: Open browser after delay
start /b cmd /c "timeout /t 8 >nul && start http://localhost:8501"

:: Run
streamlit run "%APP_FILE%" --server.headless=false

echo.
echo [✓] Session ended. Press any key to close.
pause