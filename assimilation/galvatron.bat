@echo off
REM ============================================================
REM  galvatron.bat -- run a REAL checkpoint inside leCore.
REM  Uses the SAME private venv assimilate.bat creates, so numpy
REM  (and transformers, when present) are already there. Falls back
REM  to system python only if that venv does not exist yet.
REM ============================================================
REM  USUAL ORDER:
REM    galvatron.bat work\assimilated --verify        (leCore vs reference)
REM    galvatron.bat work\original --compare work\assimilated --ppl @file.txt
REM    galvatron.bat work\assimilated --prove --doc lecore
REM    galvatron.bat work\assimilated --chat          (context persists)
REM  BUILD A BUNDLE:
REM    galvatron.bat work\assimilated --imbue work\galvatron --ban "..."
REM    ..then run it with:  run_galvatron.bat work\galvatron chat
REM ============================================================
setlocal
set "GALVATRON_CWD=%CD%"
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"
if "%~1"=="" (
  echo usage: galvatron.bat MODEL_DIR [--chat ^| --ppl TEXT ^| --prove ^| --imbue OUT ^| ...]
  "%VPY%" assimilation\galvatron.py --help
  pause
  exit /b 1
)
"%VPY%" assimilation\galvatron.py %*
if errorlevel 1 (
  echo.
  echo   [!] FAILED -- the error is printed above this line.
  pause
  exit /b 1
)
pause
