@echo off
title Satta King AI Prediction & Relation Hub
echo ========================================================
echo   Satta King AI Prediction & Relation Hub
echo   Faridabad, Ghaziabad, Gali, Desawar
echo   Live source: https://satta-king-fast.com
echo ========================================================
echo.
cd /d "%~dp0"
echo Starting application on http://127.0.0.1:8080 ...
start "" http://127.0.0.1:8080
python app.py
pause
