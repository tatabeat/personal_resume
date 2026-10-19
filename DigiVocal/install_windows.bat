@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher "py" was not found.
  echo Install Python 3.12 with the Python Launcher enabled.
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

echo Installing DigiVocal and its dependencies into .venv...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :fail

echo.
echo Installation completed. Run run_windows.bat to start the GUI.
pause
exit /b 0

:fail
echo.
echo Installation failed. Copy the error message when asking for help.
pause
exit /b 1
