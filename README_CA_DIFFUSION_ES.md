# CA-Diffusion-ES v6

v6 修正 v5 的 Drone 1 高投影率，并移除使用正式起终点训练的泄漏风险。

## 核心修改

1. 困难 replay 覆盖 Drone 0/1/5/6/8，但训练起点和终点必须分别偏移 12--35 m，禁止使用正式任务原点。
2. replay 数据集记录 `replay_excludes_exact_endpoints=true`，便于实验审计。
3. 拓扑后代的前向噪声和反向随机噪声同时缩放。
4. 若某代拓扑后代投影率超过 35%，噪声依次按 `0.25 -> 0.10 -> 0.05` 衰减。
5. history 新增 `topology_noise_scale`，可以逐代核查自适应过程。
6. 验证种子固定为 42/142/242，正式实验改用未参与验证的 1042 系列。

## 运行顺序

```powershell
cd D:\desktop\code
.\run_ca_diffusion_es.ps1 test
.\run_ca_diffusion_es.ps1 prepare
.\run_ca_diffusion_es.ps1 formal-ca-v6-check
```

分别运行三个验证种子：

```powershell
.\.venv\Scripts\python.exe -m ca_diffusion_es.cli plan --legacy-root D:\desktop\code --hgnn-csv D:\desktop\code\ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv --drones "0,1,2,5,6,8" --population 40 --generations 10 --seed 42 --output-dir D:\desktop\code\runs\ca_v6_validation_seed42
.\.venv\Scripts\python.exe -m ca_diffusion_es.cli plan --legacy-root D:\desktop\code --hgnn-csv D:\desktop\code\ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv --drones "0,1,2,5,6,8" --population 40 --generations 10 --seed 142 --output-dir D:\desktop\code\runs\ca_v6_validation_seed142
.\.venv\Scripts\python.exe -m ca_diffusion_es.cli plan --legacy-root D:\desktop\code --hgnn-csv D:\desktop\code\ALL_predicted_vs_true_pathloss_hgnn_split_5.3.csv --drones "0,1,2,5,6,8" --population 40 --generations 10 --seed 242 --output-dir D:\desktop\code\runs\ca_v6_validation_seed242
```

三个验证种子都通过后才运行：

```powershell
.\run_ca_diffusion_es.ps1 formal-ca-v6
.\run_ca_diffusion_es.ps1 report-ca-v6
```

## 新缓存

- `ca_diffusion_es\cache\legal_prior_local_context_perturbed_replay_v7_30cp.npz`
- `ca_diffusion_es\cache\trajectory_prior_local_context_perturbed_replay_v7_30cp_T25.pt`
