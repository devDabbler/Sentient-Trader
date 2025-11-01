@echo off
echo.
echo ========================================
echo   Switching to Live Trading Mode
echo ========================================
echo.
echo ⚠️  WARNING: This will use REAL MONEY!
echo.
echo Configuration Summary:
echo - $500 total capital
echo - $100 reserved (20%%)
echo - $300 active trading capital
echo - Max $75 per position
echo - Max 3 trades per day
echo - Max $10 daily loss
echo.
set /p confirm="Type 'YES' to confirm live trading: "
if /i not "%confirm%"=="YES" (
    echo.
    echo ❌ Live trading cancelled.
    pause
    exit /b
)

echo.
echo Stopping current autotrader...
call stop_autotrader.bat

echo.
echo Setting environment to Live Trading...
powershell -Command "(Get-Content .env) -replace 'IS_PAPER_TRADING=True', 'IS_PAPER_TRADING=False' -replace 'PAPER_TRADING_MODE=True', 'PAPER_TRADING_MODE=False' | Set-Content .env"

echo.
echo ✅ Switched to Live Trading Mode!
echo.
echo Configuration: config_live_trading.py
echo - $500 capital allocation
echo - Conservative risk settings
echo - 3 trades per day max
echo - 15%% position sizes
echo.
echo ⚠️  MONITOR CLOSELY! This uses real money.
echo.
echo To start trading: start_autotrader_background.bat
echo.
pause
