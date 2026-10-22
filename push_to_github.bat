@echo off
title Upload to GitHub (gauravsharma79656-web/satta-king)
cd /d "%~dp0"
echo ========================================================
echo   Uploading Satta King Project to GitHub
echo   Repository: https://github.com/gauravsharma79656-web/satta-king
echo ========================================================
echo.
echo Pushing code, templates, data, and models to GitHub...
echo If a browser window opens, please click "Authorize" to log in.
echo.
git push -u origin main
echo.
echo ========================================================
echo   All files uploaded successfully!
echo ========================================================
pause
