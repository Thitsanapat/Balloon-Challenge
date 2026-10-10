@echo off
cd /d "%~dp0"
echo BACKUP AGENT: previous guarded innovation v1, not the final-night joint agent.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m scripts.qualification_run --config "BalloonPoppingGymEnv\evaluation\configs\submission_scenario4_guarded_innovation_launch42.yaml" %*
) else (
  python -m scripts.qualification_run --config "BalloonPoppingGymEnv\evaluation\configs\submission_scenario4_guarded_innovation_launch42.yaml" %*
)
set "BPC_EXIT=%ERRORLEVEL%"
echo.
if not "%BPC_EXIT%"=="0" echo Run did not complete successfully. Read the message above before uploading anything.
pause
exit /b %BPC_EXIT%
