@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "REPO_DIR=%%~fI"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

if not defined AURA_PYTHON set "AURA_PYTHON=%REPO_DIR%\.venv\Scripts\python.exe"
if not defined INFERENCE_SYSTEM_HOST set "INFERENCE_SYSTEM_HOST=127.0.0.1"
if not defined INFERENCE_SYSTEM_PORT set "INFERENCE_SYSTEM_PORT=15880"
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
if not defined SYSTEM2_CHECK_LORA_SCALE set "SYSTEM2_CHECK_LORA_SCALE=1.0"
if not defined PLANNER_MODEL_HOST set "PLANNER_MODEL_HOST=127.0.0.1"
if not defined PLANNER_MODEL_PORT set "PLANNER_MODEL_PORT=8093"
if not defined PLANNER_MODEL_PATH set "PLANNER_MODEL_PATH=%REPO_DIR%\artifacts\models\Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined PLANNER_LLAMA_SERVER set "PLANNER_LLAMA_SERVER=%REPO_DIR%\llama.cpp\llama-server.exe"
if not defined PLANNER_PARALLEL_SLOTS set "PLANNER_PARALLEL_SLOTS=2"
if not defined DIALOGUE_MODEL_HOST set "DIALOGUE_MODEL_HOST=127.0.0.1"
if not defined DIALOGUE_MODEL_PORT set "DIALOGUE_MODEL_PORT=8094"
if not defined DIALOGUE_MODEL_PATH set "DIALOGUE_MODEL_PATH=%REPO_DIR%\artifacts\models\Qwen3-1.7B-Q4_K_M-Instruct.gguf"
if not defined DIALOGUE_LLAMA_SERVER set "DIALOGUE_LLAMA_SERVER=%REPO_DIR%\llama.cpp\llama-server.exe"
if not defined DIALOGUE_ALLOW_PROMPT_ONLY set "DIALOGUE_ALLOW_PROMPT_ONLY=0"
set "DIALOGUE_PROMPT_ONLY_ARG=--no-dialogue-allow-prompt-only"
if /I "%DIALOGUE_ALLOW_PROMPT_ONLY%"=="1" set "DIALOGUE_PROMPT_ONLY_ARG=--dialogue-allow-prompt-only"
if /I "%DIALOGUE_ALLOW_PROMPT_ONLY%"=="true" set "DIALOGUE_PROMPT_ONLY_ARG=--dialogue-allow-prompt-only"
if /I "%DIALOGUE_ALLOW_PROMPT_ONLY%"=="yes" set "DIALOGUE_PROMPT_ONLY_ARG=--dialogue-allow-prompt-only"
if /I "%DIALOGUE_ALLOW_PROMPT_ONLY%"=="on" set "DIALOGUE_PROMPT_ONLY_ARG=--dialogue-allow-prompt-only"

set "PRINT_CONFIG_JSON=0"
for %%A in (%*) do if /I "%%~A"=="-PrintConfigJson" set "PRINT_CONFIG_JSON=1"
if "%PRINT_CONFIG_JSON%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$dialogueLora = '' + $env:DIALOGUE_LORA_ADAPTER_PATH; $dialoguePromptOnly = @('1','true','yes','on') -contains ('' + $env:DIALOGUE_ALLOW_PROMPT_ONLY).Trim().ToLowerInvariant(); $system2CheckLora = '' + $env:SYSTEM2_CHECK_LORA_ADAPTER_PATH; $system2CheckPrompt = '' + $env:SYSTEM2_CHECK_SESSION_SYSTEM_PROMPT; $cfg = [ordered]@{ inference_system_host = $env:INFERENCE_SYSTEM_HOST; inference_system_port = [int]$env:INFERENCE_SYSTEM_PORT; inference_system_url = ('http://{0}:{1}' -f $env:INFERENCE_SYSTEM_HOST, $env:INFERENCE_SYSTEM_PORT); navdp_url = ('http://{0}:{1}' -f $env:NAVDP_HOST, $env:NAVDP_PORT); navdp_tensorrt_mode = $env:NAVDP_TENSORRT_MODE; navdp_tensorrt_engine_dir = $env:NAVDP_TENSORRT_ENGINE_DIR; navdp_tensorrt_precision = $env:NAVDP_TENSORRT_PRECISION; system2_url = ('http://{0}:{1}' -f $env:SYSTEM2_HOST, $env:SYSTEM2_PORT); system2_check_lora_adapter_path = $system2CheckLora; system2_check_lora_configured = -not [string]::IsNullOrWhiteSpace($system2CheckLora); system2_check_lora_scale = [double]$env:SYSTEM2_CHECK_LORA_SCALE; system2_check_session_system_prompt_configured = -not [string]::IsNullOrWhiteSpace($system2CheckPrompt); planner_model_url = ('http://{0}:{1}/v1/chat/completions' -f $env:PLANNER_MODEL_HOST, $env:PLANNER_MODEL_PORT); planner_parallel_slots = [int]$env:PLANNER_PARALLEL_SLOTS; dialogue_model_url = ('http://{0}:{1}/v1/chat/completions' -f $env:DIALOGUE_MODEL_HOST, $env:DIALOGUE_MODEL_PORT); dialogue_lora_adapter_path = $dialogueLora; dialogue_lora_configured = -not [string]::IsNullOrWhiteSpace($dialogueLora); dialogue_prompt_only = $dialoguePromptOnly; child_processes = @('navdp_model','system2','planner_llm','reasoning_dialogue') }; $cfg | ConvertTo-Json -Compress -Depth 5"
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

call "%AURA_PYTHON%" -m systems.inference.api.serve_inference_system ^
    --host "%INFERENCE_SYSTEM_HOST%" ^
    --port "%INFERENCE_SYSTEM_PORT%" ^
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
    --system2-check-lora-adapter-path "%SYSTEM2_CHECK_LORA_ADAPTER_PATH%" ^
    --system2-check-lora-scale "%SYSTEM2_CHECK_LORA_SCALE%" ^
    --system2-check-session-system-prompt "%SYSTEM2_CHECK_SESSION_SYSTEM_PROMPT%" ^
    --planner-host "%PLANNER_MODEL_HOST%" ^
    --planner-port "%PLANNER_MODEL_PORT%" ^
    --planner-model-path "%PLANNER_MODEL_PATH%" ^
    --planner-llama-server "%PLANNER_LLAMA_SERVER%" ^
    --planner-parallel-slots "%PLANNER_PARALLEL_SLOTS%" ^
    --dialogue-host "%DIALOGUE_MODEL_HOST%" ^
    --dialogue-port "%DIALOGUE_MODEL_PORT%" ^
    --dialogue-model-path "%DIALOGUE_MODEL_PATH%" ^
    --dialogue-llama-server "%DIALOGUE_LLAMA_SERVER%" ^
    --dialogue-lora-adapter-path "%DIALOGUE_LORA_ADAPTER_PATH%" ^
    %DIALOGUE_PROMPT_ONLY_ARG% ^
    %*

set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
