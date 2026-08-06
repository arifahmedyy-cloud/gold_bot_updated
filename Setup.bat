@echo off
title Gold Bot - Setup
echo ============================================
echo   Gold Bot - First Time Setup
echo ============================================
echo.
echo This will create a Python environment and install
echo everything the bot needs. This only needs to run once.
echo.
pause

cd /d "%~dp0"

echo.
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Python not found. Please install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/3] Setting up config file...
if not exist ".env" (
    copy .env.example .env >nul
    echo Created .env file - edit it with Notepad to add your MT5/API details.
)

echo.
echo ============================================
echo   Setup complete!
echo   Double-click "Run Gold Bot.bat" to start.
echo ============================================
pause
