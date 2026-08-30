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

REM SELF-HEAL FOR EXISTING VENVS (cp88): a box that already built the venv
REM before the gpu bootstrap existed gets cupy here, once, quietly.
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    REM ctk-selfheal: import succeeding is not enough -- the JIT must COMPILE.
    "%VPY%" -c "import cupy; cupy.asarray([1.,2.,3.])[cupy.asarray([0,2])]" >nul 2>&1 || (
        echo   NVIDIA GPU detected, cupy absent -- installing cupy-cuda12x...
        "%VPY%" -m pip install --quiet "cupy-cuda12x[ctk]" || echo   cupy install failed; staying on cpu numpy
    )
)

set "SRC=%~1"
set "DST=%~2"
REM THE PIPELINE'S POINT (cp95): with no argument, prefer the ASSIMILATED
REM model when it exists -- assimilate -> install = Galvatron. The old default
REM silently overrode install.py's resolution with work\original
REM (field-caught). We cd'd to the repo root above, so this exists-test is
REM correct for the DEFAULT (user-typed relative paths still resolve from the
REM caller's directory inside install.py, unchanged).
if "%SRC%"=="" (
    if exist "assimilation\work\assimilated\config.json" (
        set "SRC=work\assimilated"
        echo   [pipeline] no model given -- using work\assimilated ^(assimilate -^> install = Galvatron^)
    ) else if exist "work\assimilated\config.json" (
        set "SRC=work\assimilated"
        echo   [pipeline] no model given -- using work\assimilated ^(assimilate -^> install = Galvatron^)
    ) else (
        set "SRC=work\original"
        echo   [pipeline] work\assimilated not found -- using work\original
    )
)
REM DST is NOT defaulted here (cp96): install.py places the output BESIDE the
REM resolved model (assimilation\work\galvatron), which is only computable
REM after resolution. The old default here put it at the repo root
REM (field-caught: a stray top-level work\galvatron).
REM NO EXISTENCE CHECK HERE ON PURPOSE. This script cd'd to the repo root, so
REM testing "%SRC%" tests the WRONG directory for any relative path -- it would
REM reject a path that is perfectly correct from where the user is standing.
REM install.py resolves it properly (caller's cwd, then repo, then work\) and
REM prints every place it looked if it truly cannot find one.
REM THE ARROW WAS A REDIRECT (cp97b, field-caught: "The syntax of the command
REM is incorrect"): an unescaped > in this echo has been silently writing a
REM stray FILE named after the destination on every run; removing the DST
REM default left the redirect with no target and cmd finally said so out loud.
if "%DST%"=="" (
    echo   %SRC%  -^> [beside the resolved model]
) else (
    echo   %SRC%  -^>  %DST%
)
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
