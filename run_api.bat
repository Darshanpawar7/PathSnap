@echo off
rem PathSnap API - run this on LAPTOP 1 (needs Python + requirements.txt installed)
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
echo.
echo ===== PathSnap API (laptop 1) =====
echo.
echo On LAPTOP 2, type this address in the UI (NOT localhost):
powershell -NoProfile -Command "Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object { '      http://' + $_.IPv4Address.IPAddress + ':8000      (' + $_.InterfaceAlias + ', ' + $_.NetProfile.Name + ')' }"
echo.
echo Test it first: open that address + /health in laptop 2's browser.
echo If Windows asks about the firewall, tick Private AND Public, then click "Allow access".
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
