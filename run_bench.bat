@echo off
REM ============================================================================
REM  run_bench.bat -- benchmark leCore's GPU kernels in a THROWAWAY virtualenv.
REM
REM  Nothing is installed into your system Python. Everything lands in
REM  .venv-bench\ next to this script, which you can delete at any time.
REM
REM  Usage (from anywhere):
REM      run_bench.bat                     run it
REM      run_bench.bat --repeats 25        more samples per kernel
REM      run_bench.bat --clean             delete the venv first, then run
REM
REM  Output: a table on screen and results_<COMPUTERNAME>.json next to this file.
REM ============================================================================
setlocal EnableDelayedExpansion

set "HERE=%~dp0"
set "VENV=%HERE%.venv-bench"
set "BENCH=%HERE%research\shader_retrieval\bench_gpu.py"

if /I "%~1"=="--clean" (
    echo [clean] removing %VENV%
    if exist "%VENV%" rmdir /S /Q "%VENV%"
    shift
)

if not exist "%BENCH%" (
    echo.
    echo   ERROR: cannot find %BENCH%
    echo   Run this script from the repo root -- it expects research\shader_retrieval\ beside it.
    exit /b 1
)

REM ---- find a Python. `py -3` is the Windows launcher; fall back to python on PATH. -----------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo   ERROR: no Python found. Install Python 3.9+ from python.org and re-run.
    exit /b 1
)
echo [python] using: %PY%
%PY% --version

REM ---- create the venv only if it is missing, so re-runs are fast ------------------------------
if not exist "%VENV%\Scripts\python.exe" (
    echo [venv]   creating %VENV%
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo   ERROR: venv creation failed. On some systems you need:  %PY% -m pip install virtualenv
        exit /b 1
    )
) else (
    echo [venv]   reusing %VENV%   ^(run with --clean to rebuild^)
)

set "VPY=%VENV%\Scripts\python.exe"

REM ---- dependencies. numpy for the reference maths, moderngl+glcontext for the GL context. -----
REM      Pinned to majors, not exact versions: the point is to measure YOUR driver, not to
REM      reproduce a lockfile. --disable-pip-version-check keeps the output readable.
echo [deps]   installing numpy, moderngl (quiet)
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%VPY%" -m pip install --quiet --disable-pip-version-check "numpy>=1.24" "moderngl>=5.8" "glcontext>=2.5"
if errorlevel 1 (
    echo.
    echo   ERROR: dependency install failed. If you are offline, this script cannot proceed --
    echo   it needs numpy and moderngl from PyPI.
    exit /b 1
)

REM ---- run. PYTHONPATH lets the benchmark import the real repo corpus instead of falling back --
REM      to a synthetic one; the benchmark PRINTS which corpus it used either way.
set "PYTHONPATH=%HERE%"
set "PYTHONHASHSEED=0"
set "OUT=%HERE%results_%COMPUTERNAME%.json"

echo [run]    %BENCH%
echo.
"%VPY%" "%BENCH%" --json "%OUT%" %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo [done]   results written to %OUT%
    echo          The venv is at %VENV% -- delete it whenever you like, nothing else was touched.
) else (
    echo [failed] benchmark exited with code %RC%
)
endlocal & exit /b %RC%
