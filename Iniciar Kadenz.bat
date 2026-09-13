@echo off
rem === Lanzador de Kadenz (doble clic) ===
rem Si el servidor ya esta corriendo, solo abre el navegador.
rem Si no, lo arranca en una ventana propia minimizada y luego abre el navegador.
cd /d "%~dp0"

curl -s -o nul --max-time 3 http://127.0.0.1:8765
if %errorlevel%==0 (
    echo Kadenz ya esta en marcha. Abriendo navegador...
    start "" http://127.0.0.1:8765
    goto :eof
)

echo Arrancando Kadenz... (no cierres la ventana minimizada "Kadenz server")
start "Kadenz server" /min ".venv\Scripts\python.exe" run.py

rem Esperar a que el servidor responda (hasta ~30s; la 1a vez carga librerias de IA)
setlocal enabledelayedexpansion
set /a tries=0
:wait
ping -n 3 127.0.0.1 >nul
curl -s -o nul --max-time 3 http://127.0.0.1:8765
if !errorlevel!==0 goto :ready
set /a tries+=1
if !tries! lss 15 goto :wait
echo No se pudo confirmar el arranque. Abriendo navegador de todos modos...

:ready
start "" http://127.0.0.1:8765
endlocal
