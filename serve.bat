@echo off
REM ============================================================================
REM  serve.bat -- start the leCore standalone API service on Windows.
REM
REM  Talks to the engine over HTTP/JSON (see holographic_service.py for the API).
REM  The service is stdlib-only except for numpy, so there is almost nothing to
REM  install: this finds Python, makes sure numpy is present, and starts the
REM  server. Any extra arguments pass straight through, e.g.:
REM
REM      serve.bat                      (local only, port 8080)
REM      serve.bat --port 9000          (a different port)
REM      serve.bat --host 0.0.0.0 --token secret   (expose on the network --
REM                                     behind auth/TLS on a TRUSTED network only)
REM ============================================================================
setlocal
title leCore API service
cd /d "%~dp0"
set PYTHONHASHSEED=0

REM --- make sure Python is available ------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo   [!] Python was not found on your PATH.
    echo       Install Python 3.10+ from https://www.python.org/downloads/
    echo       and tick "Add python.exe to PATH" during setup, then re-run this.
    pause
    exit /b 1
)

REM --- make sure numpy (the only hard dependency) is available ----------------
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo   Installing numpy ^(first run only^)...
    python -m pip install numpy
    if errorlevel 1 (
        echo   [!] Could not install numpy - run:  python -m pip install numpy
        pause
        exit /b 1
    )
)

REM --- launch (pass through any --host / --port / --token) --------------------
echo   Starting the leCore API service.  Press Ctrl+C here to stop.
python holographic_service.py %*

endlocal
