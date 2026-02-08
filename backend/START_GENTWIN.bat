@echo off
REM GenTwin System - Windows Startup Script
REM Double-click this file to start both backend and frontend

echo ============================================================
echo   GenTwin System - Automated Startup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Please install Node.js
    pause
    exit /b 1
)

echo [1/4] Checking backend dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask flask-cors torch pandas numpy --break-system-packages
)

echo [2/4] Starting Backend Server...
start "GenTwin Backend" cmd /k "cd /d %~dp0 && python app_gentwin.py"

timeout /t 3 /nobreak >nul

echo [3/4] Starting Frontend Server...
cd iik-frontend-
start "GenTwin Frontend" cmd /k "npm run dev"

timeout /t 2 /nobreak >nul

echo.
echo ============================================================
echo   GenTwin System Started Successfully!
echo ============================================================
echo.
echo   Backend:  http://localhost:5000
echo   Frontend: http://localhost:5173
echo.
echo   Press any key to open browser...
echo ============================================================
pause >nul

start http://localhost:5173

echo.
echo System is running!
echo Close this window to keep both servers running.
pause
