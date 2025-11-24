' VBS script to auto-start services at login (runs hidden)
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c ""cd /d ""C:\Users\seaso\Sentient Trader"" && START_SERVICES.bat""", 0, False
