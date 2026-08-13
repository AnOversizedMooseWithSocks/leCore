@echo off
REM  audit.bat -- is the installed model actually wired, or just written?
REM      audit.bat                   looks at work\galvatron
REM      audit.bat C:\path\to\model
setlocal
REM SET BEFORE THE cd, or %CD% records the repo root and preserves nothing.
set "GALVATRON_CWD=%CD%"
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"
set "SRC=%~1"
if "%SRC%"=="" set "SRC=work\galvatron"
"%VPY%" tools\install_audit.py "%SRC%"
echo.
pause
