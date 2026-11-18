@echo off
rem PathSnap evaluator UI - run this on LAPTOP 2 (needs only Python; no packages)
cd /d "%~dp0ui"
echo.
echo ===== PathSnap UI (laptop 2) =====
echo Opening http://localhost:5500 ... then type laptop 1's address (http://ADDRESS:8000) and press Connect.
echo Press Ctrl+C to stop.
echo.
start "" http://localhost:5500
python -m http.server 5500
pause
