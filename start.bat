@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)
call "%~dp0venv\Scripts\activate.bat"
if exist "venv\Scripts\pythonw.exe" (
    start "Office Vision Tracker" "%~dp0venv\Scripts\pythonw.exe" "%~dp0main.py"
) else (
    python "%~dp0main.py"
)
