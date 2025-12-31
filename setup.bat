@echo off
echo ========================================
echo AI News Credibility Checker Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Setup complete! 
echo.
echo To start the application:
echo 1. Run start_server.bat to start the backend
echo 2. Open index.html in your web browser
echo.
echo Backend will be available at: http://localhost:8000
echo.
pause