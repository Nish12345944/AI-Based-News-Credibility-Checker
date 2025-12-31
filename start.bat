@echo off
echo Starting AI News Credibility Checker...
echo.

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Flask server...
cd backend
python app.py

pause