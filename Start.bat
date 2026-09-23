@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" beat_app.py
) else (
  py -3.13 beat_app.py
)
if errorlevel 1 pause
