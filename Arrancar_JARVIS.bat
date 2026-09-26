@echo off
title JARVIS - Mark XL
cd /d "%~dp0"

echo ========================================
echo   JARVIS - Mark XL
echo ========================================
echo.

REM Comprobar si Ollama esta corriendo
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Ollama ya esta corriendo
) else (
    echo [..] Arrancando Ollama...
    start "" /B ollama serve
    timeout /t 3 /nobreak >nul
)

echo.
echo [..] Activando entorno virtual...
call .venv\Scripts\activate

echo [..] Arrancando JARVIS...
echo.
python main.py

echo.
echo JARVIS cerrado.
pause
