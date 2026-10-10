@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m scripts.qualification_run %*
) else (
  python -m scripts.qualification_run %*
)
set "BPC_EXIT=%ERRORLEVEL%"
echo.
if not "%BPC_EXIT%"=="0" echo Run did not complete successfully. Read the message above before uploading anything.
pause
exit /b %BPC_EXIT%
