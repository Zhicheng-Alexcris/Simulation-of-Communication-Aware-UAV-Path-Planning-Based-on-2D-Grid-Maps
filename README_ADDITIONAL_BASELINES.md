# Additional formal baselines

This experiment adds two A*-free comparison methods without changing the
existing A*, A*+NSGA-II, or CA-Diffusion-ES v9 result directories.

## Methods

- `nsga2`: standard NSGA-II with full-airspace random control-point
  initialization, SBX crossover, polynomial mutation, and no A*, corridor
  projection, diffusion prior, replay, or topology repair.
- `diffusion_only`: independent full conditional DDPM samples accumulated into
  a feasible Pareto archive. It has no parent selection, truncated denoising,
  replay anchor, projection feedback, or evolutionary loop.

Both methods use the same start/end pairs, trajectory representation, Radio
Map evaluator, collision evaluator, seeds, and 4,000 black-box evaluations per
optimized UAV as the formal comparison.

## Run in VS Code PowerShell

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v9.ps1 additional-check
.\run_ca_diffusion_es_v9.ps1 additional-smoke
.\run_ca_diffusion_es_v9.ps1 additional
.\run_ca_diffusion_es_v9.ps1 report
```

The additional experiment writes to:

```text
runs\formal_experiment_additional_baselines
```

The `report` action automatically includes the additional directory after its
`records.csv` exists and writes the combined five-method figures to:

```text
runs\formal_experiment_ca_v9\report
```

The formal runner intentionally requires an empty additional output directory
so a partial run cannot be mistaken for a complete result.
