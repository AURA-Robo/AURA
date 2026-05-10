param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 18096,
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
$env:PYTHONPATH = "$repoRoot\src"

& $resolvedPython -m runtime.api.serve_runtime `
    --host $BindHost `
    --port $Port `
    --repo-root $repoRoot
