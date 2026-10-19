@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Project environment not found. Run install_windows.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "%~dp0main.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] DigiVocal exited with code %EXIT_CODE%.
  pause
)
endlocal & exit /b %EXIT_CODE%
