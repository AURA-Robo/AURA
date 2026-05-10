param(
    [string]$BackendUrl = "http://127.0.0.1:18095",
    [string]$BackendScript = "",
    [string]$DashboardRoot = "",
    [string]$Npm = "npm.cmd",
    [int]$BackendWaitSeconds = 60,
    [switch]$SkipBackendWait
)

$ErrorActionPreference = "Stop"
$resolvedBackendUrl = $BackendUrl.TrimEnd("/")
$systemRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$auraRoot = Split-Path -Parent $systemRoot
$defaultDashboardDir = Join-Path $auraRoot "dashboard"
$dashboardDir = if ([string]::IsNullOrWhiteSpace($DashboardRoot)) {
    if ([string]::IsNullOrWhiteSpace($env:AURA_DASHBOARD_ROOT)) {
        $defaultDashboardDir
    } else {
        $env:AURA_DASHBOARD_ROOT
    }
} else {
    $DashboardRoot
}
if (!(Test-Path -LiteralPath $dashboardDir -PathType Container)) {
    throw "Dashboard frontend root not found: $dashboardDir"
}
if (!(Test-Path -LiteralPath (Join-Path $dashboardDir "package.json") -PathType Leaf)) {
    throw "Dashboard frontend package.json not found: $dashboardDir"
}
if (!(Test-Path -LiteralPath (Join-Path $dashboardDir "src-tauri\\tauri.conf.json") -PathType Leaf)) {
    throw "Dashboard Tauri config not found: $dashboardDir\\src-tauri\\tauri.conf.json"
}
$null = Get-Command $Npm -ErrorAction Stop
$BackendWaitSeconds = [Math]::Max(0, $BackendWaitSeconds)
$env:AURA_DASHBOARD_API_BASE_URL = $resolvedBackendUrl
$env:AURA_DASHBOARD_PROXY_TARGET = $resolvedBackendUrl

Write-Host "[dashboard] frontend root: $dashboardDir"
Write-Host "[dashboard] backend url: $resolvedBackendUrl"

if (-not [string]::IsNullOrWhiteSpace($BackendScript)) {
    if (!(Test-Path -LiteralPath $BackendScript -PathType Leaf)) {
        throw "Backend launch script not found: $BackendScript"
    }
    Write-Host "[dashboard] starting backend script: $BackendScript"
    $backendProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $BackendScript) -PassThru
    Write-Host "[dashboard] backend process started (pid=$($backendProcess.Id))"
}

if ($SkipBackendWait) {
    Write-Host "[dashboard] skipping backend readiness check"
} elseif ($BackendWaitSeconds -le 0) {
    Write-Host "[dashboard] backend readiness wait disabled"
} else {
    $bootstrapUrl = "$resolvedBackendUrl/api/bootstrap"
    $deadline = (Get-Date).AddSeconds($BackendWaitSeconds)
    $backendReady = $false
    $attempt = 0
    Write-Host "[dashboard] waiting up to $BackendWaitSeconds seconds for backend bootstrap: $bootstrapUrl"
    while ((Get-Date) -lt $deadline) {
        $attempt += 1
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $bootstrapUrl -TimeoutSec 2 | Out-Null
            $backendReady = $true
            Write-Host "[dashboard] backend is ready"
            break
        } catch {
            if (($attempt -eq 1) -or ($attempt % 5 -eq 0)) {
                Write-Host "[dashboard] backend not ready yet; retrying..."
            }
            Start-Sleep -Seconds 1
        }
    }
    if (-not $backendReady) {
        Write-Warning "[dashboard] backend bootstrap did not respond within $BackendWaitSeconds seconds. Starting dashboard anyway."
    }
}

Push-Location $dashboardDir
try {
    Write-Host "[dashboard] starting Tauri dashboard with '$Npm run tauri:dev'"
    & $Npm run tauri:dev
} finally {
    Pop-Location
}
