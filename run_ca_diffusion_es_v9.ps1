param(
    [ValidateSet("test", "baseline-check", "formal-check", "additional-check", "all-check", "validate", "baseline", "formal", "additional", "formal-all", "report")]
    [string]$Action = "validate",
    [string]$Drones = "0,1,2,5,6,8",
    [int]$Population = 40,
    [int]$Generations = 10,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $CodeRoot ".venv\Scripts\python.exe"
$BaselineConfig = Join-Path $CodeRoot "ca_diffusion_es\formal_experiment.json"
$Config = Join-Path $CodeRoot "ca_diffusion_es\formal_experiment_ca_v9.json"
$AdditionalConfig = Join-Path $CodeRoot "ca_diffusion_es\formal_experiment_additional_baselines.json"
$BaselineOutput = Join-Path $CodeRoot "runs\formal_experiment_hv2000"
$CaOutput = Join-Path $CodeRoot "runs\formal_experiment_ca_v9_hv2000"
$AdditionalOutput = Join-Path $CodeRoot "runs\formal_experiment_additional_baselines_hv2000"
Set-Location $CodeRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found at $Python. Create .venv and install ca_diffusion_es\requirements.txt first."
}
if ($Action -eq "test") {
    & $Python -m unittest discover -s .\ca_diffusion_es\tests -v
    exit $LASTEXITCODE
}
if ($Action -eq "baseline-check") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $BaselineConfig --dry-run
    exit $LASTEXITCODE
}
if ($Action -eq "formal-check") {
    & $Python -m ca_diffusion_es.experiment_v9 --legacy-root $CodeRoot --config $Config --dry-run
    exit $LASTEXITCODE
}
if ($Action -eq "additional-check") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $AdditionalConfig --dry-run
    exit $LASTEXITCODE
}
if ($Action -eq "all-check") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $BaselineConfig --dry-run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $AdditionalConfig --dry-run
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m ca_diffusion_es.experiment_v9 --legacy-root $CodeRoot --config $Config --dry-run
    exit $LASTEXITCODE
}
if ($Action -eq "validate") {
    $Output = Join-Path $CodeRoot "runs\ca_v9_validation_seed$Seed"
    & $Python -m ca_diffusion_es.cli_v9 plan `
        --legacy-root $CodeRoot `
        --hgnn-csv (Join-Path $CodeRoot "ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv") `
        --drones $Drones --population $Population --generations $Generations `
        --seed $Seed --output-dir $Output
    exit $LASTEXITCODE
}
if ($Action -eq "baseline") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $BaselineConfig
    exit $LASTEXITCODE
}
if ($Action -eq "formal") {
    & $Python -m ca_diffusion_es.experiment_v9 --legacy-root $CodeRoot --config $Config
    exit $LASTEXITCODE
}
if ($Action -eq "additional") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $AdditionalConfig
    exit $LASTEXITCODE
}
if ($Action -eq "formal-all") {
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $BaselineConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m ca_diffusion_es.experiment --legacy-root $CodeRoot --config $AdditionalConfig
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m ca_diffusion_es.experiment_v9 --legacy-root $CodeRoot --config $Config
    exit $LASTEXITCODE
}
if ($Action -eq "report") {
    $RequiredRecords = @(
        (Join-Path $BaselineOutput "records.csv"),
        (Join-Path $AdditionalOutput "records.csv"),
        (Join-Path $CaOutput "records.csv")
    )
    foreach ($RecordsPath in $RequiredRecords) {
        if (-not (Test-Path -LiteralPath $RecordsPath)) {
            throw "Formal hv2000 result is incomplete or missing: $RecordsPath"
        }
    }
    $ReportArguments = @(
        "-m", "ca_diffusion_es.report",
        "--legacy-root", $CodeRoot,
        "--experiment-dir", $BaselineOutput,
        "--ca-rerun-dir", $CaOutput,
        "--additional-dir", $AdditionalOutput
    )
    & $Python @ReportArguments
    exit $LASTEXITCODE
}
