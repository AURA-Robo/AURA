param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 18095,
    [string]$ApiBaseUrl = "",
    [string]$DevOrigin = "http://127.0.0.1:5173",
    [string]$RuntimeUrl = "",
    [string]$InferenceSystemUrl = "",
    [string]$ReasoningSystemUrl = "",
    [string]$NavigationSystemUrl = "",
    [string]$ControlRuntimeUrl = "",
    [string]$WebRtcProxyBase = "",
    [string]$WebRtcRgbFps = "",
    [string]$WebRtcDepthFps = "",
    [string]$WebRtcTelemetryHz = "",
    [string]$WebRtcPollIntervalMs = "",
    [string]$WebRtcLatestFrameDrainBatches = "",
    [string]$WebRtcObjectMemoryQueueSize = "",
    [string]$WebRtcEnableDepthTrack = "",
    [string]$ObjectMemoryDsn = "",
    [string]$ObjectMemoryEventLogPath = "",
    [string]$ObjectMemoryAutoMigrate = "",
    [string]$KnowledgeDsn = "",
    [string]$MemoryUserId = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Resolve-SystemPython {
    param(
        [string]$RequestedPython,
        [string]$RepoRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        return $RequestedPython
    }
    if (-not [string]::IsNullOrWhiteSpace($env:AURA_PYTHON)) {
        return $env:AURA_PYTHON
    }

    return (Join-Path $RepoRoot ".venv\Scripts\python.exe")
}

function Assert-PythonCommand {
    param(
        [string]$PythonCommand,
        [string]$RepoRoot
    )

    if ((Test-Path -LiteralPath $PythonCommand -PathType Leaf) -or (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        return
    }

    $setupScript = Join-Path $RepoRoot "scripts\setup_system_venv_windows.ps1"
    throw "System Python not found: $PythonCommand. Run '$setupScript' before starting system modules."
}

$resolvedPython = Resolve-SystemPython -RequestedPython $Python -RepoRoot $repoRoot
Assert-PythonCommand -PythonCommand $resolvedPython -RepoRoot $repoRoot
$env:AURA_PYTHON = $resolvedPython

$resolvedApiBaseUrl = if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_DASHBOARD_API_BASE_URL)) {
        "http://127.0.0.1:$Port"
    } else {
        $env:AURA_DASHBOARD_API_BASE_URL
    }
} else {
    $ApiBaseUrl
}
$resolvedRuntimeUrl = if ([string]::IsNullOrWhiteSpace($RuntimeUrl)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AURA_RUNTIME_URL)) {
        $env:AURA_RUNTIME_URL
    } elseif ([string]::IsNullOrWhiteSpace($env:AURA_RUNTIME_SUPERVISOR_URL)) {
        ""
    } else {
        $env:AURA_RUNTIME_SUPERVISOR_URL
    }
} else {
    $RuntimeUrl
}
$resolvedInferenceSystemUrl = if ([string]::IsNullOrWhiteSpace($InferenceSystemUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_INFERENCE_SYSTEM_URL)) {
        "http://127.0.0.1:15880"
    } else {
        $env:AURA_INFERENCE_SYSTEM_URL
    }
} else {
    $InferenceSystemUrl
}
$resolvedReasoningSystemUrl = if ([string]::IsNullOrWhiteSpace($ReasoningSystemUrl)) {
    if (-not [string]::IsNullOrWhiteSpace($env:AURA_REASONING_SYSTEM_URL)) {
        $env:AURA_REASONING_SYSTEM_URL
    } elseif (-not [string]::IsNullOrWhiteSpace($env:AURA_PLANNER_SYSTEM_URL)) {
        $env:AURA_PLANNER_SYSTEM_URL
    } else {
        "http://127.0.0.1:17881"
    }
} else {
    $ReasoningSystemUrl
}
$resolvedNavigationSystemUrl = if ([string]::IsNullOrWhiteSpace($NavigationSystemUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_NAVIGATION_SYSTEM_URL)) {
        "http://127.0.0.1:17882"
    } else {
        $env:AURA_NAVIGATION_SYSTEM_URL
    }
} else {
    $NavigationSystemUrl
}
$resolvedControlRuntimeUrl = if ([string]::IsNullOrWhiteSpace($ControlRuntimeUrl)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_CONTROL_RUNTIME_URL)) {
        "http://127.0.0.1:8892"
    } else {
        $env:AURA_CONTROL_RUNTIME_URL
    }
} else {
    $ControlRuntimeUrl
}
$resolvedWebRtcProxyBase = if ([string]::IsNullOrWhiteSpace($WebRtcProxyBase)) {
    $env:AURA_WEBRTC_PROXY_BASE
} else {
    $WebRtcProxyBase
}
$resolvedWebRtcRgbFps = if ([string]::IsNullOrWhiteSpace($WebRtcRgbFps)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_RGB_FPS)) { "30" } else { $env:AURA_WEBRTC_RGB_FPS }
} else {
    $WebRtcRgbFps
}
$resolvedWebRtcDepthFps = if ([string]::IsNullOrWhiteSpace($WebRtcDepthFps)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_DEPTH_FPS)) { "15" } else { $env:AURA_WEBRTC_DEPTH_FPS }
} else {
    $WebRtcDepthFps
}
$resolvedWebRtcTelemetryHz = if ([string]::IsNullOrWhiteSpace($WebRtcTelemetryHz)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_TELEMETRY_HZ)) { "10" } else { $env:AURA_WEBRTC_TELEMETRY_HZ }
} else {
    $WebRtcTelemetryHz
}
$resolvedWebRtcPollIntervalMs = if ([string]::IsNullOrWhiteSpace($WebRtcPollIntervalMs)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_POLL_INTERVAL_MS)) { "10" } else { $env:AURA_WEBRTC_POLL_INTERVAL_MS }
} else {
    $WebRtcPollIntervalMs
}
$resolvedWebRtcLatestFrameDrainBatches = if ([string]::IsNullOrWhiteSpace($WebRtcLatestFrameDrainBatches)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_LATEST_FRAME_DRAIN_BATCHES)) { "8" } else { $env:AURA_WEBRTC_LATEST_FRAME_DRAIN_BATCHES }
} else {
    $WebRtcLatestFrameDrainBatches
}
$resolvedWebRtcObjectMemoryQueueSize = if ([string]::IsNullOrWhiteSpace($WebRtcObjectMemoryQueueSize)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_OBJECT_MEMORY_QUEUE_SIZE)) { "8" } else { $env:AURA_WEBRTC_OBJECT_MEMORY_QUEUE_SIZE }
} else {
    $WebRtcObjectMemoryQueueSize
}
$resolvedWebRtcEnableDepthTrack = if ([string]::IsNullOrWhiteSpace($WebRtcEnableDepthTrack)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_WEBRTC_ENABLE_DEPTH_TRACK)) { "false" } else { $env:AURA_WEBRTC_ENABLE_DEPTH_TRACK }
} else {
    $WebRtcEnableDepthTrack
}
$resolvedObjectMemoryDsn = if ([string]::IsNullOrWhiteSpace($ObjectMemoryDsn)) {
    $env:AURA_OBJECT_MEMORY_DSN
} else {
    $ObjectMemoryDsn
}
$resolvedObjectMemoryEventLogPath = if ([string]::IsNullOrWhiteSpace($ObjectMemoryEventLogPath)) {
    $env:AURA_OBJECT_MEMORY_EVENT_LOG_PATH
} else {
    $ObjectMemoryEventLogPath
}
$resolvedObjectMemoryAutoMigrate = if ([string]::IsNullOrWhiteSpace($ObjectMemoryAutoMigrate)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_OBJECT_MEMORY_AUTO_MIGRATE)) { "false" } else { $env:AURA_OBJECT_MEMORY_AUTO_MIGRATE }
} else {
    $ObjectMemoryAutoMigrate
}
$resolvedKnowledgeDsn = if ([string]::IsNullOrWhiteSpace($KnowledgeDsn)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_KNOWLEDGE_DSN)) { $resolvedObjectMemoryDsn } else { $env:AURA_KNOWLEDGE_DSN }
} else {
    $KnowledgeDsn
}
$resolvedMemoryUserId = if ([string]::IsNullOrWhiteSpace($MemoryUserId)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_MEMORY_USER_ID)) { "local-operator" } else { $env:AURA_MEMORY_USER_ID }
} else {
    $MemoryUserId
}

$env:PYTHONPATH = "$repoRoot\src"

$backendArgs = @(
    "-m", "backend.api.serve_backend",
    "--host", $BindHost,
    "--port", "$Port",
    "--api-base-url", $resolvedApiBaseUrl,
    "--dev-origin", $DevOrigin,
    "--inference-system-url", $resolvedInferenceSystemUrl,
    "--reasoning-system-url", $resolvedReasoningSystemUrl,
    "--navigation-system-url", $resolvedNavigationSystemUrl,
    "--control-runtime-url", $resolvedControlRuntimeUrl,
    "--webrtc-rgb-fps", $resolvedWebRtcRgbFps,
    "--webrtc-depth-fps", $resolvedWebRtcDepthFps,
    "--webrtc-telemetry-hz", $resolvedWebRtcTelemetryHz,
    "--webrtc-poll-interval-ms", $resolvedWebRtcPollIntervalMs,
    "--webrtc-latest-frame-drain-batches", $resolvedWebRtcLatestFrameDrainBatches,
    "--webrtc-object-memory-queue-size", $resolvedWebRtcObjectMemoryQueueSize,
    "--object-memory-user-id", $resolvedMemoryUserId
)

if (-not [string]::IsNullOrWhiteSpace($resolvedRuntimeUrl)) {
    $backendArgs += @("--runtime-url", $resolvedRuntimeUrl)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedObjectMemoryDsn)) {
    $backendArgs += @("--object-memory-dsn", $resolvedObjectMemoryDsn)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedObjectMemoryEventLogPath)) {
    $backendArgs += @("--object-memory-event-log-path", $resolvedObjectMemoryEventLogPath)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedKnowledgeDsn)) {
    $backendArgs += @("--knowledge-dsn", $resolvedKnowledgeDsn)
}

if (-not [string]::IsNullOrWhiteSpace($resolvedWebRtcProxyBase)) {
    $backendArgs += @("--webrtc-proxy-base", $resolvedWebRtcProxyBase)
}

$objectMemoryAutoMigrateEnabled = @("1", "true", "yes", "on") -contains $resolvedObjectMemoryAutoMigrate.Trim().ToLowerInvariant()
if ($objectMemoryAutoMigrateEnabled) {
    $env:AURA_OBJECT_MEMORY_AUTO_MIGRATE = "1"
    $backendArgs += "--object-memory-auto-migrate"
} else {
    $env:AURA_OBJECT_MEMORY_AUTO_MIGRATE = "0"
    $backendArgs += "--object-memory-no-auto-migrate"
}

$depthTrackEnabled = @("1", "true", "yes", "on") -contains $resolvedWebRtcEnableDepthTrack.Trim().ToLowerInvariant()
if ($depthTrackEnabled) {
    $backendArgs += "--webrtc-enable-depth-track"
} else {
    $backendArgs += "--webrtc-disable-depth-track"
}

& $resolvedPython @backendArgs
