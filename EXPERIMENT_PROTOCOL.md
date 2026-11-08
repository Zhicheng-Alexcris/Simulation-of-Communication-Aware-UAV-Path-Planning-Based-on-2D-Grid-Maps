# Formal Experiment Protocol

## 1. Frozen comparison

The main comparison contains A*, A*+NSGA-II, and CA-Diffusion-ES. A*+NSGA-II is the
primary traditional baseline. NSGA-II is always executed; no quick-feasible early return is allowed.
Both evolutionary methods use the same decision dimension, radio map, objectives, constraints,
trajectory sampling density, and black-box evaluation budget.

## 2. Paired trials

For every UAV count and trial, all methods use identical start/end pairs, altitude-layer assignment,
city geometry, radio map, outage threshold, pipe radius, and seed derivation. The default protocol uses
N in {5, 7, 9} and 10 independent trials. A trial seed is
`base_seed + 1000 * N + 50 * trial`; the per-UAV seed adds `100 * drone_id`.

## 3. Objectives and constraints

The minimization objectives are total 3D path length and mean layer-normalized HGNN path-loss cost.
Building collision count and minimum pipe clearance are inequality constraints. Multi-UAV conflicts are
resolved after geometric planning by the same time-window scheduler for every method.

## 4. Runtime accounting

Offline diffusion-prior generation and training are excluded from online planning runtime and reported
separately. Online runtime includes initial sampling/A* search, all black-box evaluations, evolutionary
updates, final Pareto selection, and time-window scheduling. RRT seed validation counts against the
CA-Diffusion-ES black-box budget.

## 5. Required metrics

The formal records contain success, path length, normalized communication cost, mean/worst/P95 predicted
path loss, outage ratio, collision count, minimum/average clearance, turn statistics, straightness,
scenario runtime, evaluation count, Pareto size, Hypervolume, RRT seed count, and scheduled start time.

## 6. Statistical reporting

Report mean and standard deviation for continuous metrics, success rate for feasibility, and paired
Wilcoxon signed-rank tests between A*+NSGA-II and CA-Diffusion-ES. Include effect size and 95% bootstrap
confidence intervals in the final paper table. Do not select only favorable seeds.

## 7. Data limitation

The current HGNN file contains one transmitter. The experiment therefore supports a single-link claim.
A multi-user worst-link claim requires exporting and rerunning with one HGNN prediction grid per transmitter.

## 8. Commands

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es.ps1 formal-check
.\run_ca_diffusion_es.ps1 formal
.\run_ca_diffusion_es.ps1 report
```