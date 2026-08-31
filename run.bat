@echo off
rem Launches the app with the project's own interpreter.
rem
rem The dependencies (PyQt5, onnxruntime, insightface) live in .venv, NOT in the
rem system Python. Running "python ui.py" from a plain prompt fails at import
rem with ModuleNotFoundError, which in a double-clicked window looks like the
rem app simply not starting. Use this instead.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No .venv found in "%cd%".
    echo.
    echo Create it once with:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "models\inswapper_128.onnx" (
    echo Models are missing. Restore them with:
    echo     .venv\Scripts\python.exe chunker.py merge
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" ui.py %*
rem Keep the window open if it fell over, so the error is readable.
if errorlevel 1 pause
