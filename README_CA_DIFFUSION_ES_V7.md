# CA-Diffusion-ES v7

v7 fixes the high projection dependence observed for Drone 1, 5 and 8 in v6.
The diffusion prior and HGNN radio map are unchanged, so `prepare` does not need
to be rerun.

## Algorithm change

Each hard-topology candidate now carries an objective-free RRT skeleton. A
truncated diffusion proposal is decoded as smooth normal-plane residuals around
that same skeleton. The residual radius is limited by local building clearance,
the 3.5 m pipe radius and a 0.5 m reserve. Communication cost is still used only
by the black-box evaluator and Pareto selection.

This keeps the same homotopy class without copying formal endpoints into the
training set. Parent-guided projection remains only as a counted last-resort
fallback.

## Run in VS Code PowerShell

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v7.ps1 test
.\run_ca_diffusion_es_v7.ps1 formal-check
```

Run the three validation seeds before any formal experiment:

```powershell
.\run_ca_diffusion_es_v7.ps1 validate -Seed 42
.\run_ca_diffusion_es_v7.ps1 validate -Seed 142
.\run_ca_diffusion_es_v7.ps1 validate -Seed 242
```

Outputs are written to `runs\ca_v7_validation_seed42`,
`runs\ca_v7_validation_seed142` and `runs\ca_v7_validation_seed242`.

Only after all validation runs have zero collision/constraint violations and
projection rates below the 35% gate should the formal command be run:

```powershell
.\run_ca_diffusion_es_v7.ps1 formal
.\run_ca_diffusion_es_v7.ps1 report
```
