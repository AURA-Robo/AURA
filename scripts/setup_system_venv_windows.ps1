param(
    [string]$Python = "python",
    [string[]]$Extras = @(),
    [switch]$SkipInstall,
    [switch]$NoWebRtc,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-InstallExtras {
    param(
        [string[]]$RequestedExtras,
        [bool]$IncludeWebRtc
    )

    $resolvedExtras = @()
    if ($IncludeWebRtc) {
        $resolvedExtras += "webrtc"
    }

    foreach ($extra in $RequestedExtras) {
        $normalized = $extra.Trim()
        if ([string]::IsNullOrWhiteSpace($normalized)) {
            continue
        }

        $alreadyIncluded = $false
        foreach ($existing in $resolvedExtras) {
            if ($existing.Equals($normalized, [StringComparison]::OrdinalIgnoreCase)) {
                $alreadyIncluded = $true
                break
            }
        }
        if (-not $alreadyIncluded) {
            $resolvedExtras += $normalized
        }
    }

    return $resolvedExtras
}

if ($Recreate -and (Test-Path -LiteralPath $venvDir)) {
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (!(Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "[venv] creating system venv: $venvDir"
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", $venvDir)
}

if (!(Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "venv python was not created: $venvPython"
}

if (-not $SkipInstall.IsPresent) {
    Push-Location $repoRoot
    $previousPythonPath = $env:PYTHONPATH
    try {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

        Write-Host "[venv] upgrading pip"
        Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")

        $effectiveExtras = @(Resolve-InstallExtras -RequestedExtras $Extras -IncludeWebRtc:(-not $NoWebRtc.IsPresent))
        $target = "."
        if ($effectiveExtras.Count -gt 0) {
            $target = ".[{0}]" -f ($effectiveExtras -join ",")
        }

        Write-Host "[venv] installing $target"
        Invoke-Checked -FilePath $venvPython -Arguments @("-m", "pip", "install", "-e", $target)
    } finally {
        if ($null -ne $previousPythonPath) {
            $env:PYTHONPATH = $previousPythonPath
        }
        Pop-Location
    }
}

Write-Host "[venv] ready: $venvPython"
