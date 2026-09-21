@echo off
REM Chạy cắt keyframe. Vd: run.bat --limit 1
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Chua cai moi truong. Chay setup.bat truoc.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0extract_keyframes.py" %*
