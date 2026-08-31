@echo off
REM ============================================================
REM  run_galvatron.bat -- run a Galvatron bundle you already built.
REM  Use THIS after assimilate.bat. It uses the assimilation venv
REM  and never collides with the repository's own run.py.
REM
REM    run_galvatron.bat                     find a bundle, then chat
REM    run_galvatron.bat info                find a bundle, show its manifest
REM    run_galvatron.bat chat
REM    run_galvatron.bat sessions
REM    run_galvatron.bat serve --port 5930
REM    run_galvatron.bat chat --tokens 512      longer replies
REM    run_galvatron.bat --bundle work\other chat     (pick one explicitly)
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set PYTHONHASHSEED=0
set "VPY=assimilation\.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"

set "BUNDLE="
if /I "%~1"=="--bundle" (
    set "BUNDLE=%~2"
    shift
    shift
)
if "%BUNDLE%"=="" (
    for /d %%D in (assimilation\work\*) do (
        if exist "%%D\galvatron.py" set "BUNDLE=%%D"
    )
)
if "!BUNDLE!"=="" (
    echo   [!] No Galvatron bundle found under assimilation\work.
    echo       Build one first:
    echo           assimilation\assimilate.bat --ban "words to forbid"
    pause
    exit /b 1
)
if not exist "!BUNDLE!\galvatron.py" (
    echo   [!] !BUNDLE! is not a Galvatron bundle ^(no galvatron.py inside^).
    pause
    exit /b 1
)
echo   bundle: !BUNDLE!
if "%~1"=="" (
    "%VPY%" "!BUNDLE!\galvatron.py" chat
) else (
    "%VPY%" "!BUNDLE!\galvatron.py" %*
)
if errorlevel 1 (
    echo.
    echo   [!] FAILED -- the error is printed above this line.
    pause
)
endlocal
