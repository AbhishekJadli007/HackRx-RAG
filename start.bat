@echo off
echo Starting NeuroFetch Application...
echo.

echo Starting Flask Backend...
start "Flask Backend" cmd /k "cd src && python app.py"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak > nul

echo Starting React Frontend...
start "React Frontend" cmd /k "npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit this window...
pause > nul 