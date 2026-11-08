# CA-Diffusion-ES v8

v8 preserves v6/v7 files and fixes the cross-seed failures found in v7.

## Search policies

- Hard-replay is training provenance. It retains conservative low-noise
  truncated diffusion, but never receives RRT skeleton residual decoding.
- Only online objective-free RRT repair seeds carry explicit skeletons.
- Each RRT skeleton owns an adaptive residual trust region. Low feasibility
  shrinks its radius; sustained high feasibility permits slow recovery.
- Online skeleton parent slots are balanced over distinct skeleton identities.
- Validation uses `patience=None`, enforcing the exact evaluation budget.

The HGNN map, trajectory dataset and diffusion checkpoint are unchanged. Do not
rerun `prepare`.

## Validation

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es_v8.ps1 test
.\run_ca_diffusion_es_v8.ps1 formal-check
.\run_ca_diffusion_es_v8.ps1 validate -Seed 42
.\run_ca_diffusion_es_v8.ps1 validate -Seed 142
.\run_ca_diffusion_es_v8.ps1 validate -Seed 242
```

The formal experiment must not start until all validation paths are safe, every
UAV reaches 400 evaluations, and projection dependence is below 35%.
