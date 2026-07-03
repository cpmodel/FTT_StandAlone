@echo off

where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate ftt >nul 2>nul
)

python run_frontend.py

if %errorlevel% neq 0 (
    py run_frontend.py
)

pause