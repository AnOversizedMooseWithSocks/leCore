@echo off
REM ============================================================
REM  leCore chat - Windows launcher (cp63)
REM  Local venv in this folder, deps installed into it, then the
REM  chat UI at http://127.0.0.1:7860  (replaces the old console)
REM ============================================================
setlocal
title leCore chat
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --quiet --disable-pip-version-check numpy flask matplotlib
set PYTHONHASHSEED=0
start "" http://127.0.0.1:7860
python chat_server.py
