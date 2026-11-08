# CA-Diffusion-ES v9

v9 adds a local trust region for hard-replay anchors. Replay candidates retain
their conservative low-noise mutation policy, but each denoised proposal is
decoded as a smooth residual around its own legal anchor. The default maximum
replay-anchor radius is 1 m and the same per-reference feasibility feedback
shrinks or slowly recovers that radius. Online RRT skeletons keep the v8
adaptive-skeleton policy unchanged.

The prior dataset, checkpoint and HGNN map are unchanged. Do not rerun
`prepare`.

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v9.ps1 test
.\run_ca_diffusion_es_v9.ps1 all-check
.\run_ca_diffusion_es_v9.ps1 validate -Seed 42
.\run_ca_diffusion_es_v9.ps1 validate -Seed 142
.\run_ca_diffusion_es_v9.ps1 validate -Seed 242
```

Do not run `formal-all` until all three validation seeds pass the safety,
fixed-budget and 35% projection gates.

## Formal hv2000 rerun

The formal comparison uses one common Hypervolume reference point
`(2000 m, 1.05)` for all five methods. New runs are written to separate
`*_hv2000` directories, so the previous v9 results are not reused or deleted.

```powershell
.\run_ca_diffusion_es_v9.ps1 all-check
.\run_ca_diffusion_es_v9.ps1 formal-all
.\run_ca_diffusion_es_v9.ps1 report
```

`formal-all` runs the three experiment groups sequentially:

1. A* and A* + NSGA-II
2. Native NSGA-II and Diffusion-only
3. CA-Diffusion-ES v9

The convergence report aggregates the five UAV histories within each trial,
then plots the mean and 95% confidence interval across ten trials. Hypervolume
is normalized by the common reference rectangle area.
