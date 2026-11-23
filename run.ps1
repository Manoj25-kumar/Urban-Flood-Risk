Start-Process "http://localhost:8501"
if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    & ".\.venv\Scripts\streamlit.exe" run app.py
} else {
    streamlit run app.py
}
