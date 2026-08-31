@echo off
REM ============================================================================
REM  assimilation\assimilate.bat -- download Qwen3.5-0.8B, run Unicron's pass,
REM  and (optionally) measure the result, on Windows. One command.
REM
REM      assimilate.bat                          download, assimilate, REPAIR,
REM                                              and build the imbued Galvatron
REM      assimilate.bat --ban "words to forbid"   ...with a ward baked in
REM      assimilate.bat --doc mydata.txt          ...grounded in YOUR data
REM      assimilate.bat --refactor 0.01           ...decomposed and rebuilt
REM                                              smaller inside a +1% budget
REM      assimilate.bat --eval                    ...and measure before/after
REM      assimilate.bat --no-imbue                weights only, no Galvatron
REM      assimilate.bat --model Qwen/Qwen3.5-2B   other sizes
REM
REM  Everything installs into a private virtual environment at
REM  assimilation\.venv (created on first run) -- your system Python is never
REM  touched, and NO Hugging Face account or token is needed: the weights are
REM  public and the download is anonymous by construction.
REM
REM  When it finishes:
REM      assimilation\run_galvatron.bat            chat with the Galvatron
REM      assimilation\galvatron.bat work\assimilated --prove --doc lecore
REM  (chat.bat is the OLD torch harness and is no longer the way in.)
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
REM GPU FOR THE INSTALL PIPELINE (cp88): the engine's cupy backend engages
REM automatically when cupy is importable (--device auto). The CUDA Toolkit is
REM NOT required -- the wheel bundles the runtime. Fallback: cpu numpy.
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo   NVIDIA GPU detected -- installing cupy for the install pipeline...
    "%VPY%" -m pip install --quiet "cupy-cuda12x[ctk]" || echo   cupy install failed; staying on cpu numpy
)
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
    echo   Next:  assimilation\run_galvatron.bat        ^(chat with it^)
    echo          assimilation\galvatron.bat assimilation\work\assimilated --prove --doc lecore
)
if errorlevel 1 (
    echo.
    echo   [!] The command above failed - the error is printed above this line.
    pause
)

endlocal
