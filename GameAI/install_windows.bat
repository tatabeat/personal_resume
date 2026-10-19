@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher was not found. Install Python 3.12.
  pause
  exit /b 1
)
py -3.12 --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.12 was not found.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto :fail
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo Installation completed. Set OPENAI_API_KEY before running.
pause
exit /b 0

:fail
echo Installation failed.
pause
exit /b 1
