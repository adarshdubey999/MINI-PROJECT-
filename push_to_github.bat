@echo off
title Push Code to GitHub (MINI-PROJECT-)
cd /d "%~dp0"
echo ========================================================
echo   PUSHING TO GITHUB: adarshdubey999/MINI-PROJECT-
echo ========================================================
echo.
echo Remote URL: https://github.com/adarshdubey999/MINI-PROJECT-.git
echo Current Branch: main
echo.
git add -A
git commit -m "Update AI Dyscalculia Screening Tool and Web Portal"
git push -u origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo --------------------------------------------------------
    echo If the remote repository has divergent commits, run:
    echo.
    echo   git push -u origin main --force
    echo --------------------------------------------------------
) else (
    echo.
    echo ========================================================
    echo   CODE SUCCESSFULLY PUSHED TO GITHUB!
    echo   View at: https://github.com/adarshdubey999/MINI-PROJECT-
    echo ========================================================
)
echo.
pause
