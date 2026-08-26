@echo off
REM ============================================================
REM  UnifiedMind console - Windows launcher
REM  Creates a LOCAL virtual environment in this folder (no admin
REM  rights needed, nothing system-wide is touched), installs the
REM  dependencies into it, and starts the web UI (unified_app.py).
REM ============================================================
setlocal
title UnifiedMind console
cd /d "%~dp0"

echo.
echo   UnifiedMind console
echo   ===================
echo.

REM --- make sure Python is available ---------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo   [!] Python was not found on your PATH.
    echo       Install Python 3.10+ from https://www.python.org/downloads/
    echo       and tick "Add python.exe to PATH" during setup, then re-run this.
    echo.
    pause
    exit /b 1
)

REM --- create a local venv on first run (avoids the system Python) ---
if not exist ".venv\Scripts\python.exe" (
    echo   Creating a local virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo   [!] Could not create the virtual environment.
        echo       Make sure the ^"venv^" module is available ^(standard with Python 3^).
        pause
        exit /b 1
    )
)

set "PY=.venv\Scripts\python.exe"

REM --- install dependencies INTO the venv ----------------------
echo   Installing dependencies into .venv ^(first run only^)...
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   [!] Dependency install failed - see the messages above.
    pause
    exit /b 1
)

REM --- launch --------------------------------------------------
echo.
echo   Starting the server at http://127.0.0.1:5000
echo   A browser window will open shortly.
echo   Leave this window open while you use the app; press Ctrl+C here to stop.
echo.
start "" http://127.0.0.1:5000

REM "%PY%" tools\unified_app.py

set PYTHONPATH=%~dp0;%PYTHONPATH%
"%PY%" "%~dp0tools\unified_app.py"

pause

endlocal
