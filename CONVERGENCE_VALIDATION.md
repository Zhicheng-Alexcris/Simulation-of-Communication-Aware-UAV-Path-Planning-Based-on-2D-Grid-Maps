# Convergence validation (not a replacement for formal results)

Run from PowerShell:

```powershell
cd D:\desktop\code
.\run_convergence_validation.ps1 all
```

Optional preflight: `check`. Optional functional check: `smoke` (three tiny
jobs; does not measure convergence). `run` runs without plotting; `report`
regenerates the validation report from saved results.

The default Python is the same `.venv\Scripts\python.exe` used by the original
experiment. If that environment cannot start, pass `-PythonExe` with the full
path of the working experiment interpreter. No packages are installed and no
prior is trained by this launcher.

## Controlled variants

1. Original v9: original compromise-based stagnation score, replay floor 0.5.
2. HV feedback: stagnation score = negative HV / (2000 * 1.05), replay floor 0.5.
3. HV feedback + replay decay: same HV feedback, replay floor may decrease
   from 0.5 to 0.2 after the existing replay mechanism is activated and the
   existing feasibility conditions are met; it recovers on poor feasibility.

The raw offspring feasibility target is 0.85 for two consecutive generations,
decay multiplier 0.8, recovery increment 0.1. Replay gating and bypass when
online RRT parents are available remain the existing v9 rules. A nonzero
replay-active count does not imply that the quota altered the selected parents;
inspect `replay_parent_quota` in history as well.

The inherited progress tolerance stays at 1e-4. In the HV variants this means
an absolute 1e-4 normalized-HV improvement beyond the last accepted best score.
The original stagnation-driven truncation mechanism is retained; no early
stopping is enabled. Final execution-trajectory selection still uses 0.65 for
communication in all three variants. Macro mutation remains disabled.

Only a progress-score hook was extracted in the common optimizer. Its default
implementation reproduces the previous calculation. Existing v9 commands use
that default and retain their previous algorithm behavior.

## Protocol

- UAV IDs 0..4 (all N=5 endpoints), validation seeds 42,142,242.
- Effective seed = validation seed + 100 * UAV ID, paired across variants.
- 3 variants * 3 seeds * 5 UAVs = 45 single-UAV planning jobs.
- Population 100, 40 generations, exactly 4000 objective evaluations per job.
- Geometry, prior, radio map, constraints and other parameters are loaded from
  `ca_diffusion_es/formal_experiment_ca_v9.json` and frozen in the manifest.
- Existing input map and checkpoint are used as-is. This validation does not
  change or relabel the map source.
- This measures independent UAV geometry planning. It does not measure the
  multi-UAV scheduling stage or compare against the green A*+NSGA-II baseline.
- Assets are loaded before timing. `runtime_s` measures the planning call,
  excluding report I/O and a separate final-metrics evaluation. Compare runtime
  within this validation, not directly against older end-to-end timing tables.

## Outputs and resumption

Output: `D:\desktop\code\runs\convergence_validation_v1`.

- `records.csv`: every completed/failed job and final metrics.
- Per-job `history.csv`: original HV, progress indicator, truncation steps,
  raw feasibility, replay activity and actual parent-quota diagnostics.
- Per-job `pareto.csv` and `trajectory.csv`: final archive objective metrics
  and selected trajectory.
- `report/summary.csv`: early/final HV, HV AUC, runtime, length and communication.
- `report/paired_differences.csv`: variant minus original, matched by seed after
  averaging the same five UAVs. Positive HV difference means improvement;
  negative runtime/length/communication-metric difference means improvement.
- `report/convergence_data.csv`: 41 points per variant, ready for plotting.
- `report/convergence.png` and `.svg`: mean convergence curves.

Curves use the same normalization, cumulative maximum and interpolation as the
existing publication report. Means first average the five UAVs within a seed,
then the three seeds. Reported standard deviations are across the three seed
means, not confidence intervals or independent counts of 15.

Rerun the same command to resume: completed jobs are skipped, failed jobs are
retried. Code/input/config/runtime hashes prevent mixing modified experiments.
If you change settings or code after starting, use a new `output_dir` in the
validation JSON. Reports reject missing or failed jobs rather than dropping
them. Successful execution with an infeasible selected path is retained with
`success=0`.

Use these three seeds to decide whether a change is worth formal validation.
They are too few to establish superiority over A*+NSGA-II. A final algorithm
change requires a new formal run with the existing held-out formal seeds.
