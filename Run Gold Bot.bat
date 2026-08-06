@echo off
title Gold Bot - Running
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run "Setup.bat" first ^(double-click it^).
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo ============================================
echo   Starting Gold Bot...
echo   Your browser will open automatically.
echo   Keep this window open while the bot runs.
echo   Close this window to stop the bot.
echo ============================================
echo.

streamlit run app.py

pause
