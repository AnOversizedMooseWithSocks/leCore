@echo off
REM ============================================================================
REM  assimilation\chat.bat -- talk to the model you just assimilated (Windows).
REM
REM      assimilation\chat.bat                (chat with the assimilated model)
REM      assimilation\chat.bat --original     (chat with the untouched original)
REM      assimilation\chat.bat --both         (same prompt to both, side by side)
REM
REM  Uses the same private venv assimilate.bat created; installs the runtime
REM  (torch + transformers) into it on first use. No accounts, no tokens.
REM ============================================================================
setlocal
title Unicron chat
cd /d "%~dp0\.."
set VPY=assimilation\.venv\Scripts\python.exe

if not exist "%VPY%" (
    echo   [!] Run assimilation\assimilate.bat first ^(it creates the venv and the model^).
    pause
    exit /b 1
)

"%VPY%" -c "import torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo   Installing torch + transformers into the venv ^(one-time, large^)...
    REM CUDA WHEEL WHEN A GPU EXISTS (cp87): plain `pip install torch` on
    REM Windows ships the CPU-ONLY build -- an A4500 would sit idle. Detect
    REM nvidia-smi and install the cu124 wheel; fall back to CPU torch.
    where nvidia-smi >nul 2>&1
    if %errorlevel%==0 (
        echo   NVIDIA GPU detected -- installing CUDA torch...
        "%VPY%" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124 || "%VPY%" -m pip install --quiet torch
    ) else (
        "%VPY%" -m pip install --quiet torch
    )
    "%VPY%" -m pip install --quiet transformers
)

"%VPY%" assimilation\chat.py %*
if errorlevel 1 (
    echo.
    echo   [!] The command above failed - the error is printed above this line.
    pause
)

endlocal
