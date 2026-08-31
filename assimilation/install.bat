@echo off
REM ============================================================
REM  install.bat -- Unicron makes Galvatron. One step, no pipeline.
REM
REM    install.bat                          work\original  -> work\galvatron
REM    install.bat MODEL_DIR                MODEL_DIR      -> work\galvatron
REM    install.bat MODEL_DIR OUT_DIR        wherever you like
REM
REM  This REPLACES assimilate -> repair -> imbue. That path edited 18 of 265
REM  tensors, repair reverted 12 of them as harmful, and what survived sat
REM  inside the measurement noise. NOTHING HERE EDITS YOUR ORIGINAL TENSORS.
REM  Two blank layers go in FRONT, and everything leCore adds lives in them,
REM  in vocabulary rows your tokenizer never emits, or in reserved directions
REM  of the recurrent state.
REM
REM  Optional, only if you want them:
REM    --doc FILE       your own text (default: leCore's own documentation)
REM    --registers N    permanent memory slots (default: model width / 8)
REM    --passages N     searchable passages (default: as many rows as are free)
REM ============================================================
setlocal
REM KEEP THE CALLER'S DIRECTORY. We cd to the repo root so the package
REM imports work, which otherwise breaks every RELATIVE path the user types:
REM "install.bat models\qwen" would look under the repo, not under where they
REM are standing, and no argument can correct that. galvatron.bat has always
REM done this; install.bat did not.
set "GALVATRON_CWD=%CD%"
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"

set "SRC=%~1"
set "DST=%~2"
if "%SRC%"=="" set "SRC=work\original"
if "%DST%"=="" set "DST=work\galvatron"
REM NO EXISTENCE CHECK HERE ON PURPOSE. This script cd'd to the repo root, so
REM testing "%SRC%" tests the WRONG directory for any relative path -- it would
REM reject a path that is perfectly correct from where the user is standing.
REM install.py resolves it properly (caller's cwd, then repo, then work\) and
REM prints every place it looked if it truly cannot find one.
echo   %SRC%  ->  %DST%
echo.
if "%~2"=="" (
  "%VPY%" assimilation\install.py "%SRC%" "%DST%" %2 %3 %4 %5 %6 %7
) else (
  "%VPY%" assimilation\install.py %*
)
if errorlevel 1 (
  echo.
  echo   [!] FAILED -- the error is printed above this line.
  pause
  exit /b 1
)
echo.
echo   Next:  assess.bat
pause
