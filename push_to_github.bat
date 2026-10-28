@echo off
title Push Code to GitHub (screening-tool-)
cd /d "%~dp0"
echo ========================================================
echo   PUSHING TO GITHUB: adarshdubey999/screening-tool-
echo ========================================================
echo.
echo Remote URL: https://github.com/adarshdubey999/screening-tool-.git
echo Current Branch: main
echo.
git push -u origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo --------------------------------------------------------
    echo If the remote repository already contains files (like a README or LICENSE),
    echo run this command to force push the initial code:
    echo.
    echo   git push -u origin main --force
    echo --------------------------------------------------------
) else (
    echo.
    echo ========================================================
    echo   CODE SUCCESSFULLY PUSHED TO GITHUB!
    echo   View at: https://github.com/adarshdubey999/screening-tool-
    echo ========================================================
)
echo.
pause
