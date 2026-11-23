@echo off
title FloodSight Web App
echo Starting FloodSight Web Application...
start http://localhost:8501
if exist ".venv\Scripts\streamlit.exe" (
    .\.venv\Scripts\streamlit.exe run app.py
) else (
    streamlit run app.py
)
pause
