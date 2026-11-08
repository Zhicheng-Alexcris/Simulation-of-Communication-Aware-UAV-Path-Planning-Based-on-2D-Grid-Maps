param(
    [ValidateSet("check", "test", "run")]
    [string]$Action = "check",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ObjectiveRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $ObjectiveRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python not found: $PythonExe. Supply -PythonExe with your experiment interpreter."
}
$ObjectiveConfig = Join-Path $ObjectiveRoot "ca_diffusion_es\formal_experiment_ca_v9_objective_logs.json"
Push-Location $ObjectiveRoot
try {
    if ($Action -eq "test") {
        & $PythonExe -m unittest ca_diffusion_es.tests.test_objective_logging -v
    } elseif ($Action -eq "check") {
        & $PythonExe -m ca_diffusion_es.experiment_v9 --legacy-root $ObjectiveRoot --config $ObjectiveConfig --dry-run
    } else {
        & $PythonExe -m ca_diffusion_es.experiment_v9 --legacy-root $ObjectiveRoot --config $ObjectiveConfig
    }
    $ObjectiveExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $ObjectiveExitCode
