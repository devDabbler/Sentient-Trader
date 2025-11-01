@echo off
echo.
echo ========================================
echo   Switching to Paper Trading Mode
echo ========================================
echo.

echo Stopping current autotrader...
call stop_autotrader.bat

echo.
echo Setting environment to Paper Trading...
powershell -Command "(Get-Content .env) -replace 'IS_PAPER_TRADING=False', 'IS_PAPER_TRADING=True' -replace 'PAPER_TRADING_MODE=False', 'PAPER_TRADING_MODE=True' | Set-Content .env"

echo.
echo ✅ Switched to Paper Trading Mode!
echo.
echo Configuration: config_paper_trading.py
echo - Full paper account balance
echo - Aggressive settings for testing
echo - 15 trades per day max
echo - 5%% position sizes
echo.
echo To start trading: start_autotrader_background.bat
echo.
pause
