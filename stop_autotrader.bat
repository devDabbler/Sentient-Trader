@echo off
REM Stop Auto-Trader Background Process

echo.
echo ========================================
echo   Stopping Auto-Trader Background
echo ========================================
echo.

REM Kill pythonw.exe processes (background auto-trader)
echo Stopping pythonw.exe processes (auto-trader)...
taskkill /F /IM pythonw.exe 2>nul

if %errorlevel% == 0 (
    echo Auto-trader process stopped!
) else (
    echo No pythonw.exe processes found
)

REM Also check for regular python.exe running the script
echo.
echo Checking for python.exe running auto-trader...
wmic process where "commandline like '%%run_autotrader_background.py%%'" delete 2>nul

echo.
echo Done! Auto-trader should be stopped.
echo Check Task Manager to verify no python/pythonw processes remain.

echo.
pause

