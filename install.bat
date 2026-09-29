@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================
echo  Office Vision Tracker - installation
echo ============================================
echo.

set "BASE_PY="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=py -3"
if defined BASE_PY goto :have_python
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "BASE_PY=python"
:have_python
if defined BASE_PY goto :create_venv
echo Python 3.9+ was not found.
echo Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/
echo Enable the Python launcher, or add python.exe to PATH.
goto :failed
:create_venv
echo Using:
%BASE_PY% --version
echo.
if exist "venv\Scripts\python.exe" goto :venv_ready
echo Creating virtual environment...
%BASE_PY% -m venv venv
if errorlevel 1 goto :failed
:venv_ready
set "PY=%~dp0venv\Scripts\python.exe"
echo Updating pip...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
echo Installing dependencies...
"%PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"
if not exist "screenshots" mkdir "screenshots"
echo Downloading MediaPipe models...
"%PY%" "%~dp0download_models.py"
if errorlevel 1 goto :failed
echo Checking imports...
"%PY%" -c "import PySide6, cv2, mediapipe, numpy; print('PySide6', PySide6.__version__); print('OpenCV', cv2.__version__); print('MediaPipe', mediapipe.__version__); print('NumPy', numpy.__version__)"
if errorlevel 1 goto :failed
echo.
echo Installation completed successfully.
if /I "%~1"=="nopause" goto :eof
pause
exit /b 0
:failed
echo.
echo Installation failed. See the messages above.
if /I "%~1"=="nopause" exit /b 1
pause
exit /b 1
