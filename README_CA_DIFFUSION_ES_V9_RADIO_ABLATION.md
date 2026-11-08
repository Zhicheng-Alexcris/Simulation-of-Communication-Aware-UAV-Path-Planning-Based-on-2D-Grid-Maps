# CA-Diffusion-ES v9 Radio Map Paired Ablation

This experiment isolates the planning-time Radio Map input while keeping the
UAV, random seed, diffusion prior, optimizer parameters, and 4,000-evaluation
budget fixed. Both final trajectories are evaluated using `True_PathLoss`;
`Predicted_PathLoss` is available only to the `With Radio Map` planner.

The formal protocol contains 5 UAVs x 10 seeds x 2 paired variants. Existing
formal CA v9 trajectories are reused for the `With Radio Map` arm, so only the
50 `Without Radio Map` trajectories require new optimization.

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v9_radio_ablation.ps1 check
.\run_ca_diffusion_es_v9_radio_ablation.ps1 smoke
.\run_ca_diffusion_es_v9_radio_ablation.ps1 smoke-report
.\run_ca_diffusion_es_v9_radio_ablation.ps1 run
.\run_ca_diffusion_es_v9_radio_ablation.ps1 report
```

The formal output is written to `runs/v9_radio_map_ablation`. The main paper
figure is `report/radio_map_ablation_true_channel.png`; paired statistics are
stored in `report/radio_map_ablation_summary.csv`.

The runner is resumable. Use `-RerunFailed` only after inspecting a failed
run's `traceback.txt`.
