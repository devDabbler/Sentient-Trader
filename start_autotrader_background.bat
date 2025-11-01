@echo off
REM Start Auto-Trader in Background on Windows
REM This keeps it running even after you close the command prompt

echo.
echo ========================================
echo   Starting Auto-Trader Background
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start with pythonw (no console window)
echo Starting auto-trader in background...
start /B pythonw run_autotrader_background.py

echo.
echo ✅ Auto-trader started in background!
echo.
echo To stop it, run: stop_autotrader.bat
echo Or find "pythonw.exe" in Task Manager and end it
echo.
echo Logs: logs\autotrader_background.log
echo.

pause

