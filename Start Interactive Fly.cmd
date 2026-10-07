@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run scripts\setup.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m virtual_fly live
if errorlevel 1 pause
