@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run install_windows.bat first.
  pause
  exit /b 1
)
if not defined OPENAI_API_KEY (
  echo [ERROR] OPENAI_API_KEY is not set in this environment.
  echo See README.md for instructions. Never put a real key in this file.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" main.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
