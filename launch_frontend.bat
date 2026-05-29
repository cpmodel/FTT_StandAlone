@echo off
setlocal

where conda >nul 2>nul
if %errorlevel%==0 (
    call conda activate ftt >nul 2>nul
)

set "PY_CMD="
where python >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    echo Could not find a Python executable on PATH.
    echo Install Python or activate the correct environment, then try again.
    pause
    exit /b 1
)

%PY_CMD% run_frontend.py

pause