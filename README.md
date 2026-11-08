# CA-Diffusion-ES Package

该包的正式实验入口和当前算法口径见：

- `..\README_CA_DIFFUSION_ES.md`
- `..\EXPERIMENT_PROTOCOL.md`
- `formal_experiment.json`

关键模块：

- `radio_map.py`：HGNN 三维 Radio Map 与最差链路聚合；
- `prior_data.py`、`diffusion_model.py`：无 A* 轨迹先验；
- `optimizer.py`：固定评价预算的双目标 truncated Diffusion-ES；
- `baselines.py`：统一口径的 A* 和 A*+NSGA-II；
- `evaluation.py`：所有方法共享的黑盒评价；
- `experiment.py`：正式批量实验；
- `report.py`：正式图表和统计检验。

不要使用旧 `main.py --compare`、旧 `artifacts` 或旧结果表。正式命令为：

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es.ps1 formal-check
.\run_ca_diffusion_es.ps1 formal
.\run_ca_diffusion_es.ps1 report
```