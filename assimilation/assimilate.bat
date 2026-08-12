@echo off
REM ============================================================================
REM  assimilation\assimilate.bat -- download Qwen3.5-0.8B. Spectral filtering
REM  is retained only as an explicit research control.
REM
REM      assimilate.bat                          download untouched weights
REM      assimilate.bat --research-spectral --eval
REM                                              run the research control
REM      assimilate.bat --model Qwen/Qwen3.5-2B   other sizes
REM
REM  Everything installs into a private virtual environment at
REM  assimilation\.venv (created on first run) -- your system Python is never
REM  touched, and NO Hugging Face account or token is needed: the weights are
REM  public and the download is anonymous by construction.
REM
REM  See assimilation\README.md before opting into spectral filtering or the
REM  separate experimental layer-prepending installer.
REM ============================================================================
setlocal
title Unicron assimilation
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set HF_HUB_DISABLE_TELEMETRY=1

REM --- make sure Python is available ------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo   [!] Python was not found on your PATH.
    echo       Install Python 3.10+ from https://www.python.org/downloads/
    echo       and tick "Add python.exe to PATH" during setup, then re-run this.
    pause
    exit /b 1
)

REM --- private virtual environment (first run only) ----------------------------
set VPY=assimilation\.venv\Scripts\python.exe
if not exist "%VPY%" (
    echo   Creating virtual environment at assimilation\.venv ^(first run only^)...
    python -m venv assimilation\.venv
    if errorlevel 1 (
        echo   [!] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

REM --- dependencies: numpy + huggingface_hub always ----------------------------
"%VPY%" -c "import numpy, huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo   Installing numpy + huggingface_hub into the venv...
    "%VPY%" -m pip install --quiet --upgrade pip
    "%VPY%" -m pip install --quiet numpy huggingface_hub
)

REM --- torch stack only when --eval was asked for (caller-side instrument) -----
echo %* | findstr /C:"--eval" >nul
if not errorlevel 1 (
    "%VPY%" -c "import torch, transformers" >nul 2>&1
    if errorlevel 1 (
        echo   Installing torch + transformers for --eval ^(one-time, large^)...
        "%VPY%" -m pip install --quiet torch transformers
    )
)

REM --- run ---------------------------------------------------------------------
"%VPY%" assimilation\run.py --workdir assimilation\work %*
if not errorlevel 1 (
    echo.
    echo   Untouched weights: assimilation\work\original
    echo   See assimilation\README.md for explicitly gated experiment paths.
)
if errorlevel 1 (
    echo.
    echo   [!] The command above failed - the error is printed above this line.
    pause
)

endlocal
