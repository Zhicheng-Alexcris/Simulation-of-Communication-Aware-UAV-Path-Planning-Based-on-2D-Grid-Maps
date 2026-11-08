param(
    [ValidateSet("check", "smoke", "run", "report", "all")]
    [string]$Action = "all",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ValidationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $ValidationRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python not found: $PythonExe. Use -PythonExe with the interpreter used by your original experiment."
}
$ValidationConfig = Join-Path $ValidationRoot "ca_diffusion_es\convergence_validation.json"
Push-Location $ValidationRoot
try {
    & $PythonExe -c "import numpy, torch, scipy, matplotlib"
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment Python could not start or dependencies are missing. Use -PythonExe with your working experiment interpreter."
    }
    & $PythonExe -m ca_diffusion_es.convergence_validation $Action --legacy-root $ValidationRoot --config $ValidationConfig
    $ValidationExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $ValidationExitCode
