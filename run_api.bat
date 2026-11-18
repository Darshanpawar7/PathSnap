@echo off
rem PathSnap API - run this on LAPTOP 1 (needs Python + requirements.txt installed)
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
echo.
echo ===== PathSnap API (laptop 1) =====
echo Give laptop 2 one of these addresses, as  http://ADDRESS:8000
ipconfig | findstr /R /C:"IPv4"
echo.
echo If Windows asks about the firewall, click "Allow access".
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
