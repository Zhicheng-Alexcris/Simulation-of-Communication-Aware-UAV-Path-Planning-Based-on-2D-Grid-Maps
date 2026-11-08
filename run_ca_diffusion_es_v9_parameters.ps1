param(
    [ValidateSet("check", "smoke", "run", "report", "all", "smoke-report")]
    [string]$Action = "check",
    [switch]$RerunFailed
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Config = Join-Path $CodeRoot "ca_diffusion_es\parameter_experiment_v9.json"
$Output = Join-Path $CodeRoot "runs\v9_parameter_control"
$SmokeOutput = Join-Path $CodeRoot "runs\v9_parameter_control_smoke"
Set-Location $CodeRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found. Run .\run_ca_diffusion_es.ps1 setup first."
}

$RunArguments = @(
    "-m", "ca_diffusion_es.parameter_experiment_v9",
    "--legacy-root", $CodeRoot,
    "--config", $Config
)
if ($RerunFailed) {
    $RunArguments += "--rerun-failed"
}

if ($Action -eq "check") {
    & $Python @RunArguments --dry-run
    exit $LASTEXITCODE
}
if ($Action -eq "smoke") {
    & $Python @RunArguments --smoke
    exit $LASTEXITCODE
}
if ($Action -eq "run" -or $Action -eq "all") {
    & $Python @RunArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Action -eq "smoke-report") {
    & $Python -m ca_diffusion_es.parameter_report_v9 `
        --legacy-root $CodeRoot --config $Config --experiment-dir $SmokeOutput
    exit $LASTEXITCODE
}
if ($Action -eq "report" -or $Action -eq "all") {
    & $Python -m ca_diffusion_es.parameter_report_v9 `
        --legacy-root $CodeRoot --config $Config --experiment-dir $Output
    exit $LASTEXITCODE
}
