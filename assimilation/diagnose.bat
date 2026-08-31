@echo off
REM ============================================================
REM  diagnose.bat -- print the facts an install decision depends on.
REM
REM  Run this from the assimilation folder when an install fails:
REM      diagnose.bat                     looks at work\original
REM      diagnose.bat work\galvatron      or any other model
REM      diagnose.bat C:\path\to\model
REM
REM  It prints layers, dtypes, architecture family, GDN head geometry,
REM  per-layer tensor families, the prepend drift, and WHICH tensors in a
REM  blank layer are nonzero -- which is enough to locate an install failure
REM  without a round trip.
REM ============================================================
setlocal
REM SET BEFORE THE cd, or %CD% records the repo root and preserves nothing.
set "GALVATRON_CWD=%CD%"
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"

set "SRC=%~1"
if "%SRC%"=="" set "SRC=work\original"
"%VPY%" tools\diagnose_install.py "%SRC%"
if errorlevel 1 (
  echo.
  echo   [!] FAILED -- the error is printed above this line.
)
echo.
pause
