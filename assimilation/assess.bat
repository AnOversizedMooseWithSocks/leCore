@echo off
REM ============================================================
REM  assess.bat -- measure every model this pipeline produced and
REM  write one assessment bundle each, ready to send.
REM
REM    assess.bat                    measure all of work\*
REM    assess.bat work\galvatron     measure just one
REM
REM  Each bundle is a PROFILE (BIOS, perplexity, tokens/sec, gates,
REM  spectra, activations, top-64 logits, harden audit) -- NOT the
REM  model. No weight tensors travel.
REM ============================================================
setlocal enabledelayedexpansion
REM SET BEFORE THE cd, not after -- capturing %CD% once we have already
REM changed directory records the repo root and preserves nothing. Same fix as
REM install.bat, and I got the ORDER wrong here on the first attempt.
set "GALVATRON_CWD=%CD%"
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"
set "OUT=assimilation\assessments"
if not exist "%OUT%" mkdir "%OUT%"
if not "%~1"=="" (
    for %%N in ("%~1") do set "NAME=%%~nxN"
    "%VPY%" assimilation\galvatron.py "%~1" --assess "%CD%\%OUT%\!NAME!.npz"
    echo.
    echo   Bundle written to %OUT% -- send that file.
    pause
    exit /b 0
)
for /d %%D in (assimilation\work\*) do (
    if exist "%%D\model.safetensors" (
        echo.
        echo === %%~nxD ===
        "%VPY%" assimilation\galvatron.py "%%D" --assess "%CD%\%OUT%\%%~nxD.npz"
    )
)
echo.
echo   All bundles are in %OUT% -- send the whole folder.
echo   They are PROFILES, not models: no weight tensors travel.
pause
endlocal
