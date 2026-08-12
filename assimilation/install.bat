@echo off
REM ============================================================
REM  install.bat -- Unicron makes Galvatron. One step, no pipeline.
REM
REM    install.bat --experimental                          use defaults
REM    install.bat --experimental MODEL_DIR                choose source
REM    install.bat --experimental MODEL_DIR OUT_DIR        choose both
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

REM Do not pre-parse paths here. install.py owns defaults and resolves relative
REM paths against GALVATRON_CWD before the repo root.
"%VPY%" assimilation\install.py %*
if errorlevel 1 (
  echo.
  echo   [!] FAILED -- the error is printed above this line.
  pause
  exit /b 1
)
echo.
echo   Next:  assess.bat
pause
