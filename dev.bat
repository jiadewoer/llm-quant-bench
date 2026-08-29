@echo off
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo [llm-quant-bench] .venv not found. Run: uv venv
    cmd /k
    exit /b 1
)
call "%~dp0.venv\Scripts\activate.bat"
cd /d "%~dp0"
echo [llm-quant-bench] UTF-8 on, venv active, cwd = %CD%
cmd /k
