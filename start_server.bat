@echo off
rem Arranca SOLO el servidor de Kadenz (sin abrir navegador), con log.
rem Lo usan la tarea de inicio y el arranque manual; sobrevive a otras sesiones.
cd /d "%~dp0"
rem A clone has no data/ — the redirect below fails without it.
if not exist "data" mkdir "data"
".venv\Scripts\python.exe" run.py > "data\server_run.log" 2>&1
