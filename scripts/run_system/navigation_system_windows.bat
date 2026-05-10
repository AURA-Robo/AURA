@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined AURA_PYTHON set "AURA_PYTHON=%REPO_DIR%\.venv\Scripts\python.exe"
if not defined NAVIGATION_SYSTEM_HOST set "NAVIGATION_SYSTEM_HOST=127.0.0.1"
if not defined NAVIGATION_SYSTEM_PORT set "NAVIGATION_SYSTEM_PORT=17882"
if not defined SYSTEM2_URL set "SYSTEM2_URL=http://127.0.0.1:15801"
if not defined NAVDP_URL set "NAVDP_URL=http://127.0.0.1:18888"
if not defined NAVIGATION_BACKEND_AUTOSTART set "NAVIGATION_BACKEND_AUTOSTART=1"
if not defined NAVIGATION_NAVDP_FALLBACK set "NAVIGATION_NAVDP_FALLBACK=heuristic"
if not defined NAVIGATION_SYSTEM2_TIMEOUT set "NAVIGATION_SYSTEM2_TIMEOUT=20.0"
if not defined NAVIGATION_NAVDP_TIMEOUT set "NAVIGATION_NAVDP_TIMEOUT=5.0"
if not defined NAVDP_HOST set "NAVDP_HOST=127.0.0.1"
if not defined NAVDP_PORT set "NAVDP_PORT=18888"
if not defined NAVDP_CHECKPOINT set "NAVDP_CHECKPOINT=%REPO_DIR%\artifacts\models\navdp-cross-modal.ckpt"
if not defined NAVDP_DEVICE set "NAVDP_DEVICE=cuda:0"
if not defined NAVDP_TENSORRT_MODE set "NAVDP_TENSORRT_MODE=auto"
if not defined NAVDP_TENSORRT_ENGINE_DIR set "NAVDP_TENSORRT_ENGINE_DIR=%REPO_DIR%\artifacts\models\navdp_tensorrt"
if not defined NAVDP_TENSORRT_PRECISION set "NAVDP_TENSORRT_PRECISION=fp16"
if not defined SYSTEM2_HOST set "SYSTEM2_HOST=127.0.0.1"
if not defined SYSTEM2_PORT set "SYSTEM2_PORT=15801"
if not defined SYSTEM2_LLAMA_URL set "SYSTEM2_LLAMA_URL=http://127.0.0.1:15802"
if not defined SYSTEM2_MODEL_PATH set "SYSTEM2_MODEL_PATH="

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ navigation_system_host = $env:NAVIGATION_SYSTEM_HOST; navigation_system_port = [int]$env:NAVIGATION_SYSTEM_PORT; navigation_system_url = ('http://{0}:{1}' -f $env:NAVIGATION_SYSTEM_HOST, $env:NAVIGATION_SYSTEM_PORT); system2_url = $env:SYSTEM2_URL; navdp_url = $env:NAVDP_URL; navdp_fallback = $env:NAVIGATION_NAVDP_FALLBACK; navdp_tensorrt_mode = $env:NAVDP_TENSORRT_MODE; navdp_tensorrt_engine_dir = $env:NAVDP_TENSORRT_ENGINE_DIR; navdp_tensorrt_precision = $env:NAVDP_TENSORRT_PRECISION; system2_timeout = [double]$env:NAVIGATION_SYSTEM2_TIMEOUT; navdp_timeout = [double]$env:NAVIGATION_NAVDP_TIMEOUT; backend_autostart = ($env:NAVIGATION_BACKEND_AUTOSTART -ne '0') }; $cfg | ConvertTo-Json -Compress -Depth 5"
    exit /b 0
)

if not exist "%AURA_PYTHON%" (
    where %AURA_PYTHON% >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] System venv python not found: %AURA_PYTHON%
        echo [ERROR] Run "%REPO_DIR%\scripts\setup_system_venv_windows.ps1" before starting system modules.
        exit /b 1
    )
)

pushd "%REPO_DIR%"
set "PYTHONPATH=%REPO_DIR%\src"
set "BACKEND_AUTOSTART_FLAG=--backend-autostart"
if "%NAVIGATION_BACKEND_AUTOSTART%"=="0" set "BACKEND_AUTOSTART_FLAG=--no-backend-autostart"

call "%AURA_PYTHON%" -m systems.navigation.api.serve_navigation_system ^
    --host "%NAVIGATION_SYSTEM_HOST%" ^
    --port "%NAVIGATION_SYSTEM_PORT%" ^
    --system2-url "%SYSTEM2_URL%" ^
    --navdp-url "%NAVDP_URL%" ^
    %BACKEND_AUTOSTART_FLAG% ^
    --navdp-fallback "%NAVIGATION_NAVDP_FALLBACK%" ^
    --system2-timeout "%NAVIGATION_SYSTEM2_TIMEOUT%" ^
    --navdp-timeout "%NAVIGATION_NAVDP_TIMEOUT%" ^
    --navdp-host "%NAVDP_HOST%" ^
    --navdp-port "%NAVDP_PORT%" ^
    --navdp-checkpoint "%NAVDP_CHECKPOINT%" ^
    --navdp-device "%NAVDP_DEVICE%" ^
    --navdp-tensorrt-mode "%NAVDP_TENSORRT_MODE%" ^
    --navdp-tensorrt-engine-dir "%NAVDP_TENSORRT_ENGINE_DIR%" ^
    --navdp-tensorrt-precision "%NAVDP_TENSORRT_PRECISION%" ^
    --system2-host "%SYSTEM2_HOST%" ^
    --system2-port "%SYSTEM2_PORT%" ^
    --system2-llama-url "%SYSTEM2_LLAMA_URL%" ^
    --system2-model-path "%SYSTEM2_MODEL_PATH%" ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
