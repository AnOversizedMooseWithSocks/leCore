@echo off
REM ============================================================================
REM  build_pages.bat -- rebuild the browser demo pages in a THROWAWAY virtualenv,
REM  then open the main one in your browser.
REM
REM  Nothing is installed into your system Python. Everything lands in
REM  .venv-bench\ next to this script (shared with run_bench.bat), which you can
REM  delete at any time.
REM
REM  Usage (from the repo root):
REM      build_pages.bat              rebuild all five pages, then open the search page
REM      build_pages.bat --no-open    rebuild only, don't launch a browser
REM      build_pages.bat --clean      delete the venv first, then rebuild
REM      build_pages.bat --open-only  skip the build, just open what is in pages\
REM
REM  The pages are CHECKED IN under pages\ -- you only need this if you changed a
REM  shader, the corpus, or the retrieval policy and want the pages to match.
REM ============================================================================
setlocal EnableDelayedExpansion

set "HERE=%~dp0"
set "VENV=%HERE%.venv-bench"
set "GEN=%HERE%research\shader_retrieval"
set "OUT=%HERE%pages"
set "DOOPEN=1"
set "DOBUILD=1"

:parseargs
if /I "%~1"=="--clean" (
    echo [clean] removing %VENV%
    if exist "%VENV%" rmdir /S /Q "%VENV%"
    shift & goto parseargs
)
if /I "%~1"=="--no-open"   ( set "DOOPEN=0" & shift & goto parseargs )
if /I "%~1"=="--open-only" ( set "DOBUILD=0" & shift & goto parseargs )

if not exist "%OUT%" mkdir "%OUT%"

if "%DOBUILD%"=="0" goto openpage

if not exist "%GEN%\make_search_page.py" (
    echo.
    echo   ERROR: cannot find %GEN%\make_search_page.py
    echo   Run this from the repo root -- it expects research\shader_retrieval\ beside it.
    exit /b 1
)

REM ---- find a Python: `py -3` is the Windows launcher, else python on PATH -------------------
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY (
    echo.
    echo   ERROR: no Python found. Install Python 3.9+ from python.org and re-run.
    echo   ^(Or just open pages\lecore_search_webgl2.html -- it is checked in.^)
    exit /b 1
)
echo [python] using: %PY%

if not exist "%VENV%\Scripts\python.exe" (
    echo [venv]   creating %VENV%
    %PY% -m venv "%VENV%"
    if errorlevel 1 ( echo   ERROR: venv creation failed. & exit /b 1 )
) else (
    echo [venv]   reusing %VENV%   ^(--clean to rebuild^)
)
set "VPY=%VENV%\Scripts\python.exe"

REM ---- numpy for the maths, moderngl for three of the five generators (they import
REM      glsl_hier, which builds a GL context to validate the tier walk before embedding it).
REM      Installing numpy alone made those three fail INSIDE the venv while passing on a host
REM      that happened to have moderngl -- a venv missing a dep the host has is a false green.
echo [deps]   installing numpy, moderngl (quiet)
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
"%VPY%" -m pip install --quiet --disable-pip-version-check "numpy>=1.24" "moderngl>=5.8" "glcontext>=2.5"
if errorlevel 1 ( echo   ERROR: dependency install failed ^(offline?^). & exit /b 1 )

set "PYTHONPATH=%HERE%"
set "PYTHONHASHSEED=0"

REM ---- The generators MUST run with the repo root as the working directory: they write into the
REM      CWD, the corpus loader globs relative to it, and several import engine modules by their
REM      FLAT name, which only resolves from the root. Build there, then move the output.
set "FAILED="
for %%G in (make_search_page make_webgl_page make_webgl_full make_webgl_typed make_webgl_vsa) do (
    echo [build]  %%G.py
    "%VPY%" "%GEN%\%%G.py"
    if errorlevel 1 set "FAILED=!FAILED! %%G"
)
move /Y "%HERE%lecore_search_webgl2.html" "%OUT%\" >nul 2>&1
move /Y "%HERE%lecore_webgl2*.html" "%OUT%\" >nul 2>&1

echo.
if defined FAILED (
    echo [warn]   these generators failed:!FAILED!
    echo          The checked-in pages in %OUT% are still valid; only the ones above are stale.
) else (
    echo [done]   all pages rebuilt in %OUT%
)
dir /B "%OUT%\*.html"

:openpage
if "%DOOPEN%"=="1" (
    echo.
    echo [open]   %OUT%\lecore_search_webgl2.html
    start "" "%OUT%\lecore_search_webgl2.html"
    echo          If it opens in Edge, right-click the file and choose Open with ^> Chrome.
    echo          Look at the PASS table: SCORER IN USE, scatter==full scan, and the 60/60 rows.
)
endlocal & exit /b 0
