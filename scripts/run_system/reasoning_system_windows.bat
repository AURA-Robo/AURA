@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined AURA_PYTHON set "AURA_PYTHON=%REPO_DIR%\.venv\Scripts\python.exe"
if not defined REASONING_SYSTEM_HOST set "REASONING_SYSTEM_HOST=127.0.0.1"
if not defined REASONING_SYSTEM_PORT set "REASONING_SYSTEM_PORT=17881"
if not defined NAVIGATION_SYSTEM_URL set "NAVIGATION_SYSTEM_URL=http://127.0.0.1:17882"
if not defined PLANNER_MODEL_BASE_URL set "PLANNER_MODEL_BASE_URL=http://127.0.0.1:8093/v1/chat/completions"
if not defined PLANNER_MODEL_NAME set "PLANNER_MODEL_NAME=Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined PLANNER_TIMEOUT set "PLANNER_TIMEOUT=120.0"
if not defined PLANNER_INTENT_SLOT_ID set "PLANNER_INTENT_SLOT_ID=0"
if not defined PLANNER_TASK_FRAME_SLOT_ID set "PLANNER_TASK_FRAME_SLOT_ID=1"
if not defined DIALOGUE_MODEL_BASE_URL set "DIALOGUE_MODEL_BASE_URL=http://127.0.0.1:8094/v1/chat/completions"
if not defined DIALOGUE_MODEL_NAME set "DIALOGUE_MODEL_NAME=Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined DIALOGUE_TIMEOUT set "DIALOGUE_TIMEOUT=30.0"
if not defined AURA_OBJECT_MEMORY_DSN set "AURA_OBJECT_MEMORY_DSN="
if not defined AURA_OBJECT_MEMORY_AUTO_MIGRATE set "AURA_OBJECT_MEMORY_AUTO_MIGRATE=0"
if not defined AURA_MEMORY_USER_ID set "AURA_MEMORY_USER_ID=local-operator"
if not defined AURA_KNOWLEDGE_DSN set "AURA_KNOWLEDGE_DSN=%AURA_OBJECT_MEMORY_DSN%"
if not defined AURA_CONVERSATION_MEMORY_DSN set "AURA_CONVERSATION_MEMORY_DSN=%AURA_OBJECT_MEMORY_DSN%"
if not defined AURA_SCENE_PRESET set "AURA_SCENE_PRESET="

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ reasoning_system_host = $env:REASONING_SYSTEM_HOST; reasoning_system_port = [int]$env:REASONING_SYSTEM_PORT; reasoning_system_url = ('http://{0}:{1}' -f $env:REASONING_SYSTEM_HOST, $env:REASONING_SYSTEM_PORT); navigation_system_url = $env:NAVIGATION_SYSTEM_URL; planner_model_base_url = $env:PLANNER_MODEL_BASE_URL; planner_model = $env:PLANNER_MODEL_NAME; planner_timeout = [double]$env:PLANNER_TIMEOUT; planner_intent_slot_id = [int]$env:PLANNER_INTENT_SLOT_ID; planner_task_frame_slot_id = [int]$env:PLANNER_TASK_FRAME_SLOT_ID; dialogue_model_base_url = $env:DIALOGUE_MODEL_BASE_URL; dialogue_model = $env:DIALOGUE_MODEL_NAME; dialogue_timeout = [double]$env:DIALOGUE_TIMEOUT; object_memory_dsn_configured = -not [string]::IsNullOrWhiteSpace($env:AURA_OBJECT_MEMORY_DSN); object_memory_auto_migrate = @('1','true','yes','on') -contains ('' + $env:AURA_OBJECT_MEMORY_AUTO_MIGRATE).Trim().ToLowerInvariant(); knowledge_dsn_configured = -not [string]::IsNullOrWhiteSpace($env:AURA_KNOWLEDGE_DSN); conversation_memory_dsn_configured = -not [string]::IsNullOrWhiteSpace($env:AURA_CONVERSATION_MEMORY_DSN); scene_preset = $env:AURA_SCENE_PRESET; memory_user_id = $env:AURA_MEMORY_USER_ID }; $cfg | ConvertTo-Json -Compress -Depth 5"
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
set "OBJECT_MEMORY_AUTO_MIGRATE_FLAG=--object-memory-no-auto-migrate"
if /I "%AURA_OBJECT_MEMORY_AUTO_MIGRATE%"=="1" set "OBJECT_MEMORY_AUTO_MIGRATE_FLAG=--object-memory-auto-migrate"
if /I "%AURA_OBJECT_MEMORY_AUTO_MIGRATE%"=="true" set "OBJECT_MEMORY_AUTO_MIGRATE_FLAG=--object-memory-auto-migrate"
if /I "%AURA_OBJECT_MEMORY_AUTO_MIGRATE%"=="yes" set "OBJECT_MEMORY_AUTO_MIGRATE_FLAG=--object-memory-auto-migrate"
if /I "%AURA_OBJECT_MEMORY_AUTO_MIGRATE%"=="on" set "OBJECT_MEMORY_AUTO_MIGRATE_FLAG=--object-memory-auto-migrate"

call "%AURA_PYTHON%" -m systems.reasoning.api.serve_reasoning_system ^
    --host "%REASONING_SYSTEM_HOST%" ^
    --port "%REASONING_SYSTEM_PORT%" ^
    --navigation-url "%NAVIGATION_SYSTEM_URL%" ^
    --planner-model-base-url "%PLANNER_MODEL_BASE_URL%" ^
    --planner-model "%PLANNER_MODEL_NAME%" ^
    --planner-timeout "%PLANNER_TIMEOUT%" ^
    --planner-intent-slot-id "%PLANNER_INTENT_SLOT_ID%" ^
    --planner-task-frame-slot-id "%PLANNER_TASK_FRAME_SLOT_ID%" ^
    --dialogue-model-base-url "%DIALOGUE_MODEL_BASE_URL%" ^
    --dialogue-model "%DIALOGUE_MODEL_NAME%" ^
    --dialogue-timeout "%DIALOGUE_TIMEOUT%" ^
    --object-memory-dsn "%AURA_OBJECT_MEMORY_DSN%" ^
    --memory-user-id "%AURA_MEMORY_USER_ID%" ^
    %OBJECT_MEMORY_AUTO_MIGRATE_FLAG% ^
    --knowledge-dsn "%AURA_KNOWLEDGE_DSN%" ^
    --conversation-memory-dsn "%AURA_CONVERSATION_MEMORY_DSN%" ^
    --knowledge-scene-scope "%AURA_SCENE_PRESET%" ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
