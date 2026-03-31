@echo off
setlocal EnableExtensions

REM Go to the folder where this BAT file lives
cd /d "%~dp0"

REM Change this if your Python file has a different name
set "APP_FILE=survey_analyzer.py"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

title Survey Analyzer

echo === Survey Analyzer ===
echo.

if not exist "%APP_FILE%" (
    echo ERROR: Cannot find "%APP_FILE%" in:
    echo %CD%
    pause
    exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH.
        echo Please install Python from https://www.python.org
        pause
        exit /b 1
    )
    set "PY_CMD=python"
) else (
    set "PY_CMD=py"
)

if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :setup_error
)

echo Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :setup_error

if exist "requirements.txt" (
    "%VENV_PY%" -m pip install -r requirements.txt
) else (
    "%VENV_PY%" -m pip install numpy pandas scikit-learn PySide6 matplotlib shap openpyxl xlrd
)
if errorlevel 1 goto :setup_error

echo Starting application...
"%VENV_PY%" "%APP_FILE%"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" goto :run_error

endlocal
exit /b 0

:setup_error
echo.
echo Setup failed.
pause
endlocal
exit /b 1

:run_error
echo.
echo Application exited with error code %EXITCODE%.
pause
endlocal
exit /b %EXITCODE%