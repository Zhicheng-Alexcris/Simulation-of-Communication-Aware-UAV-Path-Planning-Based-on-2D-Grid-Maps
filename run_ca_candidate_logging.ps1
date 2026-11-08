param(
    [ValidateSet("check", "test", "run")]
    [string]$Action = "check",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$CandidateRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $CandidateRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python not found: $PythonExe. Supply -PythonExe with your experiment interpreter."
}
$CandidateConfig = Join-Path $CandidateRoot "ca_diffusion_es\candidate_logging_n5.json"
Push-Location $CandidateRoot
try {
    if ($Action -eq "test") {
        Write-Host "TEST ONLY: no formal experiment will run."
        & $PythonExe -m unittest ca_diffusion_es.tests.test_candidate_logging -v
    } elseif ($Action -eq "check") {
        Write-Host "CHECK ONLY: validating files and settings; no experiment will run."
        & $PythonExe -m ca_diffusion_es.experiment_v9 --legacy-root $CandidateRoot --config $CandidateConfig --dry-run
    } else {
        Write-Host "RUN: U=5, 10 repeats, 50 UAV jobs, 4000 candidate evaluations per UAV job."
        Write-Host "Output: runs\ca_v9_candidate_logs_n5_v1"
        & $PythonExe -u -m ca_diffusion_es.experiment_v9 --legacy-root $CandidateRoot --config $CandidateConfig
    }
    $CandidateExitCode = $LASTEXITCODE
    if ($CandidateExitCode -ne 0) {
        Write-Host "FAILED (exit code $CandidateExitCode). Read the error above; the experiment is not complete."
    } elseif ($Action -eq "run") {
        Write-Host "COMPLETE: candidate CSV files and generation histories have been saved."
    } elseif ($Action -eq "check") {
        Write-Host "CHECK PASSED. To start the experiment: .\run_ca_candidate_logging.ps1 -Action run"
    }
} finally {
    Pop-Location
}
exit $CandidateExitCode
