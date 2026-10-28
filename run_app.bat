@echo off
title AI-Based Dyscalculia Screening Tool - Streamlit Server
cd /d "%~dp0"
echo ========================================================
echo   AI-BASED DYSCALCULIA SCREENING TOOL
echo   Tech Stack: Python + Streamlit + Pandas + NumPy + Scikit-Learn + Matplotlib + SQLite
echo   Project Report by Aadarsh Dubey (Rizvi College of ASC)
echo   Guide: Prof. Javed Pathan
echo ========================================================
echo Starting Streamlit Application at http://localhost:8501
echo.
python -m streamlit run app.py --server.port=8501 --server.headless=false
pause
