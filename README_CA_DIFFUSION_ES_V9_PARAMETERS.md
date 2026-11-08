# CA-Diffusion-ES v9 single-parameter experiment

This experiment is independent of the formal A*/NSGA-II comparison and does not
modify `runs/formal_experiment_ca_v9`.

## Protocol

- Drones (zero-based code IDs): `0,2,3,4,8` (paper labels: 1,3,4,5,9)
- Seeds: `42,142,242`
- Budget: 4,000 black-box evaluations per single-UAV run
- Eight unique configurations, 120 single-UAV runs in total
- One-factor comparisons:
  - truncation: fixed `t=1`, fixed `t=3`, fixed `t=6`, adaptive `1/3/6`
  - prior anchor ratio: `0`, `0.25`, `0.50`, `0.75`, `1.00`

The v9 configuration (adaptive `1/3/6`, anchor ratio `0.50`) is shared by both
comparisons and is executed only once.

## Run in VS Code PowerShell

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v9_parameters.ps1 check
.\run_ca_diffusion_es_v9_parameters.ps1 smoke
.\run_ca_diffusion_es_v9_parameters.ps1 smoke-report
.\run_ca_diffusion_es_v9_parameters.ps1 all
```

`run` and `all` resume completed runs automatically. To retry only failed runs:

```powershell
.\run_ca_diffusion_es_v9_parameters.ps1 run -RerunFailed
```

Generate figures again without rerunning optimization:

```powershell
.\run_ca_diffusion_es_v9_parameters.ps1 report
```

## Outputs

Full results are written to `runs/v9_parameter_control`. Important report files:

- `report/parameter_summary.csv`
- `report/parameter_recommendation.json`
- `report/truncation_hv_convergence.png`
- `report/anchor_hv_convergence.png`
- `report/truncation_final_feasible_hv.png`
- `report/anchor_final_feasible_hv.png`
- `report/truncation_raw_feasibility.png`
- `report/truncation_projection_rate.png`
- `report/anchor_initial_feasibility.png`

The primary metric is relative feasible hypervolume at 4,000 evaluations.
Failed/infeasible runs receive zero. Convergence curves use the same fixed
black-box evaluation budget and are normalized per drone/seed pair so that one
geometrically large scenario cannot dominate the aggregate.
