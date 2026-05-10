@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined ISAACSIM_PATH set "ISAACSIM_PATH=C:\isaac-sim"
set "ISAACSIM_PYTHON=%ISAACSIM_PATH%\python.bat"
if not defined RUNTIME_CONTROL_API_HOST set "RUNTIME_CONTROL_API_HOST=127.0.0.1"
if not defined RUNTIME_CONTROL_API_PORT set "RUNTIME_CONTROL_API_PORT=8892"
if not defined NAVIGATION_URL set "NAVIGATION_URL=http://127.0.0.1:17882"
if not defined AURA_LAUNCH_MODE set "AURA_LAUNCH_MODE=gui"
if not defined AURA_VIEWER_ENABLED set "AURA_VIEWER_ENABLED=1"
if not defined AURA_MEMORY_STORE set "AURA_MEMORY_STORE=0"
if not defined AURA_DETECTION_ENABLED set "AURA_DETECTION_ENABLED=1"
if not defined AURA_DETECTION_MODEL_PATH set "AURA_DETECTION_MODEL_PATH=%REPO_DIR%\artifacts\models\yoloe-26s-seg-pf.pt"
if not defined AURA_SCENE_PRESET set "AURA_SCENE_PRESET="
if not defined AURA_KNOWLEDGE_DSN set "AURA_KNOWLEDGE_DSN=%AURA_OBJECT_MEMORY_DSN%"
if not defined CAMERA_PITCH_DEG set "CAMERA_PITCH_DEG=0.0"
if not defined AURA_ACTION_SCALE set "AURA_ACTION_SCALE=0.5"
if not defined AURA_ONNX_DEVICE set "AURA_ONNX_DEVICE=auto"
if not defined AURA_CMD_MAX_VX set "AURA_CMD_MAX_VX=0.5"
if not defined AURA_CMD_MAX_VY set "AURA_CMD_MAX_VY=0.3"
if not defined AURA_CMD_MAX_WZ set "AURA_CMD_MAX_WZ=0.8"
if not defined SCENE_USD set "SCENE_USD="
if not defined ENV_URL set "ENV_URL=/Isaac/Environments/Simple_Warehouse/warehouse.usd"
if not defined POLICY set "POLICY=%REPO_DIR%\artifacts\models\g1_policy_fp16.engine"
if not defined ROBOT_USD set "ROBOT_USD=%REPO_DIR%\robots\g1\g1_d455.usd"

if not defined SCENE_USD (
    if /I "%AURA_SCENE_PRESET%"=="interioragent" set "SCENE_USD=%REPO_DIR%\datasets\InteriorAgent\kujiale_0004\kujiale_0004_navila_sanitized.usda"
    if /I "%AURA_SCENE_PRESET%"=="interior agent kujiale 3" set "SCENE_USD=%REPO_DIR%\datasets\InteriorAgent\kujiale_0003\kujiale_0003.usda"
)

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$viewerEnabled = @('1','true','yes','on') -contains ('' + $env:AURA_VIEWER_ENABLED).ToLowerInvariant(); $memoryStore = @('1','true','yes','on') -contains ('' + $env:AURA_MEMORY_STORE).ToLowerInvariant(); $detectionEnabled = @('1','true','yes','on') -contains ('' + $env:AURA_DETECTION_ENABLED).ToLowerInvariant(); $cfg = [ordered]@{ runtime_control_api_host = $env:RUNTIME_CONTROL_API_HOST; runtime_control_api_port = [int]$env:RUNTIME_CONTROL_API_PORT; runtime_control_api_url = ('http://{0}:{1}' -f $env:RUNTIME_CONTROL_API_HOST, $env:RUNTIME_CONTROL_API_PORT); navigation_url = $env:NAVIGATION_URL; launch_mode = $env:AURA_LAUNCH_MODE; scene_preset = $env:AURA_SCENE_PRESET; scene_usd = $env:SCENE_USD; env_url = $env:ENV_URL; viewer_enabled = $viewerEnabled; viewer_publish = $viewerEnabled; memory_store = $memoryStore; detection_enabled = $detectionEnabled; detection_model_path = $env:AURA_DETECTION_MODEL_PATH; action_scale = [double]$env:AURA_ACTION_SCALE; onnx_device = $env:AURA_ONNX_DEVICE; cmd_max_vx = [double]$env:AURA_CMD_MAX_VX; cmd_max_vy = [double]$env:AURA_CMD_MAX_VY; cmd_max_wz = [double]$env:AURA_CMD_MAX_WZ; camera_pitch_deg = [double]$env:CAMERA_PITCH_DEG; knowledge_dsn_configured = -not [string]::IsNullOrWhiteSpace($env:AURA_KNOWLEDGE_DSN) }; $cfg | ConvertTo-Json -Compress -Depth 5"
    exit /b 0
)

if not exist "%ISAACSIM_PYTHON%" (
    echo [ERROR] Isaac Sim python.bat not found: %ISAACSIM_PYTHON%
    exit /b 1
)
if defined SCENE_USD if not exist "%SCENE_USD%" (
    echo [ERROR] Scene USD not found: %SCENE_USD%
    exit /b 1
)

set "HEADLESS_FLAG="
if /I "%AURA_LAUNCH_MODE%"=="headless" set "HEADLESS_FLAG=--headless"
set "VIEWER_FLAG=--viewer-publish"
if /I "%AURA_VIEWER_ENABLED%"=="0" set "VIEWER_FLAG=--no-viewer-publish"
if /I "%AURA_VIEWER_ENABLED%"=="false" set "VIEWER_FLAG=--no-viewer-publish"
if /I "%AURA_VIEWER_ENABLED%"=="off" set "VIEWER_FLAG=--no-viewer-publish"
if /I "%AURA_VIEWER_ENABLED%"=="no" set "VIEWER_FLAG=--no-viewer-publish"
set "DETECTION_FLAG=--detection-enabled"
if /I "%AURA_DETECTION_ENABLED%"=="0" set "DETECTION_FLAG=--no-detection-enabled"
if /I "%AURA_DETECTION_ENABLED%"=="false" set "DETECTION_FLAG=--no-detection-enabled"
if /I "%AURA_DETECTION_ENABLED%"=="off" set "DETECTION_FLAG=--no-detection-enabled"
if /I "%AURA_DETECTION_ENABLED%"=="no" set "DETECTION_FLAG=--no-detection-enabled"
set "SCENE_USD_FLAG="
if defined SCENE_USD set SCENE_USD_FLAG=--scene_usd "%SCENE_USD%"
set "CONFIG_DIR_FLAG="
if defined CONFIG_DIR set CONFIG_DIR_FLAG=--config_dir "%CONFIG_DIR%"

pushd "%REPO_DIR%"
set "PYTHONPATH=%REPO_DIR%\src"

call "%ISAACSIM_PYTHON%" -m systems.control.api.play_g1_internvla_navdp ^
    --policy "%POLICY%" ^
    %CONFIG_DIR_FLAG% ^
    --robot_usd "%ROBOT_USD%" ^
    --env_url "%ENV_URL%" ^
    %SCENE_USD_FLAG% ^
    --navigation_url "%NAVIGATION_URL%" ^
    --runtime_control_api_host "%RUNTIME_CONTROL_API_HOST%" ^
    --runtime_control_api_port "%RUNTIME_CONTROL_API_PORT%" ^
    --camera_api_port "0" ^
    --camera_pitch_deg "%CAMERA_PITCH_DEG%" ^
    %VIEWER_FLAG% ^
    %DETECTION_FLAG% ^
    --detection-model-path "%AURA_DETECTION_MODEL_PATH%" ^
    --action_scale "%AURA_ACTION_SCALE%" ^
    --onnx_device "%AURA_ONNX_DEVICE%" ^
    --vx_max "%AURA_CMD_MAX_VX%" ^
    --vy_max "%AURA_CMD_MAX_VY%" ^
    --wz_max "%AURA_CMD_MAX_WZ%" ^
    %HEADLESS_FLAG% ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
