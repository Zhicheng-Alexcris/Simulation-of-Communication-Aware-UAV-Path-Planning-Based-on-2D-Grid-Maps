param(
    [ValidateSet("check", "smoke", "smoke-report", "run", "report", "all")]
    [string]$Action = "check",
    [switch]$RerunFailed
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$Config = Join-Path $CodeRoot "ca_diffusion_es\radio_map_ablation_v9.json"
$Output = Join-Path $CodeRoot "runs\v9_radio_map_ablation"
$SmokeOutput = Join-Path $CodeRoot "runs\v9_radio_map_ablation_smoke"
Set-Location $CodeRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found. Run the project setup first."
}

$RunArguments = @(
    "-m", "ca_diffusion_es.radio_map_ablation_v9",
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
if ($Action -eq "smoke-report") {
    & $Python -m ca_diffusion_es.radio_map_ablation_report_v9 `
        --config $Config --experiment-dir $SmokeOutput --allow-incomplete
    exit $LASTEXITCODE
}
if ($Action -eq "run" -or $Action -eq "all") {
    & $Python @RunArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($Action -eq "report" -or $Action -eq "all") {
    & $Python -m ca_diffusion_es.radio_map_ablation_report_v9 `
        --config $Config --experiment-dir $Output
    exit $LASTEXITCODE
}
