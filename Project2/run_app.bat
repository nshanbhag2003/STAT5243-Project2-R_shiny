@echo off
REM STAT5243 Project 2 - Run Shiny App
REM Use port 8001 to avoid conflict with port 8000
echo Starting Shiny app at http://127.0.0.1:8001
echo Press Ctrl+C to stop.
python -m shiny run app --port 8001
pause
