param(
    [string]$BackendBindHost = "127.0.0.1",
    [int]$BackendPort = 18095,
    [string]$BackendUrl = "",
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
    [string]$WebRtcEnableDepthTrack = "",
    [string]$ObjectMemoryDsn = "",
    [string]$KnowledgeDsn = "",
    [string]$MemoryUserId = "",
    [string]$Python = "",
    [string]$DashboardRoot = "",
    [string]$Npm = "npm.cmd",
    [int]$BackendWaitSeconds = 60,
    [switch]$SkipBackendWait,
    [switch]$KeepBackendRunning,
    [string]$BackendScriptPath = "",
    [string]$DashboardScriptPath = ""
)

$ErrorActionPreference = "Stop"
$systemRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$resolvedBackendScriptPath = if ([string]::IsNullOrWhiteSpace($BackendScriptPath)) {
    Join-Path $PSScriptRoot "backend_windows.ps1"
} else {
    $BackendScriptPath
}
$resolvedDashboardScriptPath = if ([string]::IsNullOrWhiteSpace($DashboardScriptPath)) {
    Join-Path $PSScriptRoot "dashboard_dev_windows.ps1"
} else {
    $DashboardScriptPath
}

function Test-BackendBootstrap {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$($Url.TrimEnd('/'))/api/bootstrap" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    $completed = Start-Process -FilePath "taskkill.exe" -ArgumentList @("/PID", "$ProcessId", "/T", "/F") -NoNewWindow -Wait -PassThru
    if ($completed.ExitCode -notin @(0, 128)) {
        throw "taskkill failed for pid=$ProcessId with exit code $($completed.ExitCode)"
    }
}

if (!(Test-Path -LiteralPath $resolvedDashboardScriptPath -PathType Leaf)) {
    throw "Dashboard launcher not found: $resolvedDashboardScriptPath"
}

$resolvedBackendUrl = if (-not [string]::IsNullOrWhiteSpace($BackendUrl)) {
    $BackendUrl.TrimEnd("/")
} elseif (-not [string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
    $ApiBaseUrl.TrimEnd("/")
} else {
    $dashboardHost = if ($BackendBindHost -eq "0.0.0.0") { "127.0.0.1" } else { $BackendBindHost }
    "http://$dashboardHost`:$BackendPort"
}
$env:AURA_DASHBOARD_API_BASE_URL = $resolvedBackendUrl
$env:AURA_DASHBOARD_PROXY_TARGET = $resolvedBackendUrl

$backendOwned = $false
$backendProcess = $null

if (Test-BackendBootstrap -Url $resolvedBackendUrl) {
    Write-Host "[run_dashboard] backend already reachable at $resolvedBackendUrl; reusing existing backend"
} else {
    if (!(Test-Path -LiteralPath $resolvedBackendScriptPath -PathType Leaf)) {
        throw "Backend launcher not found: $resolvedBackendScriptPath"
    }

    $logDir = Join-Path $systemRoot "logs\run_dashboard"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $backendStdoutLog = Join-Path $logDir "backend.stdout.log"
    $backendStderrLog = Join-Path $logDir "backend.stderr.log"

    $backendArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $resolvedBackendScriptPath,
        "-BindHost", $BackendBindHost,
        "-Port", "$BackendPort"
    )
    if (-not [string]::IsNullOrWhiteSpace($Python)) { $backendArgs += @("-Python", $Python) }

    if (-not [string]::IsNullOrWhiteSpace($ApiBaseUrl)) { $backendArgs += @("-ApiBaseUrl", $ApiBaseUrl) }
    if (-not [string]::IsNullOrWhiteSpace($DevOrigin)) { $backendArgs += @("-DevOrigin", $DevOrigin) }
    if (-not [string]::IsNullOrWhiteSpace($RuntimeUrl)) { $backendArgs += @("-RuntimeUrl", $RuntimeUrl) }
    if (-not [string]::IsNullOrWhiteSpace($InferenceSystemUrl)) { $backendArgs += @("-InferenceSystemUrl", $InferenceSystemUrl) }
    if (-not [string]::IsNullOrWhiteSpace($ReasoningSystemUrl)) { $backendArgs += @("-ReasoningSystemUrl", $ReasoningSystemUrl) }
    if (-not [string]::IsNullOrWhiteSpace($NavigationSystemUrl)) { $backendArgs += @("-NavigationSystemUrl", $NavigationSystemUrl) }
    if (-not [string]::IsNullOrWhiteSpace($ControlRuntimeUrl)) { $backendArgs += @("-ControlRuntimeUrl", $ControlRuntimeUrl) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcProxyBase)) { $backendArgs += @("-WebRtcProxyBase", $WebRtcProxyBase) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcRgbFps)) { $backendArgs += @("-WebRtcRgbFps", $WebRtcRgbFps) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcDepthFps)) { $backendArgs += @("-WebRtcDepthFps", $WebRtcDepthFps) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcTelemetryHz)) { $backendArgs += @("-WebRtcTelemetryHz", $WebRtcTelemetryHz) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcPollIntervalMs)) { $backendArgs += @("-WebRtcPollIntervalMs", $WebRtcPollIntervalMs) }
    if (-not [string]::IsNullOrWhiteSpace($WebRtcEnableDepthTrack)) { $backendArgs += @("-WebRtcEnableDepthTrack", $WebRtcEnableDepthTrack) }
    if (-not [string]::IsNullOrWhiteSpace($ObjectMemoryDsn)) { $backendArgs += @("-ObjectMemoryDsn", $ObjectMemoryDsn) }
    if (-not [string]::IsNullOrWhiteSpace($KnowledgeDsn)) { $backendArgs += @("-KnowledgeDsn", $KnowledgeDsn) }
    if (-not [string]::IsNullOrWhiteSpace($MemoryUserId)) { $backendArgs += @("-MemoryUserId", $MemoryUserId) }

    Write-Host "[run_dashboard] starting backend via $resolvedBackendScriptPath"
    Write-Host "[run_dashboard] backend logs: $backendStdoutLog"
    $backendProcess = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $backendArgs `
        -WorkingDirectory $systemRoot `
        -RedirectStandardOutput $backendStdoutLog `
        -RedirectStandardError $backendStderrLog `
        -WindowStyle Hidden `
        -PassThru
    $backendOwned = $true
    Write-Host "[run_dashboard] backend process started (pid=$($backendProcess.Id))"
}

try {
    $dashboardArgs = @{
        BackendUrl = $resolvedBackendUrl
        DashboardRoot = $DashboardRoot
        Npm = $Npm
        BackendWaitSeconds = $BackendWaitSeconds
    }
    if ($SkipBackendWait) {
        $dashboardArgs["SkipBackendWait"] = $true
    }

    Write-Host "[run_dashboard] launching dashboard via $resolvedDashboardScriptPath"
    & $resolvedDashboardScriptPath @dashboardArgs
} finally {
    if ($backendOwned -and -not $KeepBackendRunning.IsPresent -and $backendProcess -ne $null) {
        try {
            Write-Host "[run_dashboard] stopping backend process tree (pid=$($backendProcess.Id))"
            Stop-ProcessTree -ProcessId $backendProcess.Id
        } catch {
            Write-Warning "[run_dashboard] failed to stop backend process tree: $($_.Exception.Message)"
        }
    }
}
