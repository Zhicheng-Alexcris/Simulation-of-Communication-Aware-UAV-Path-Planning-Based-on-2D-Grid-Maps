# 逐代目标值日志

在 PowerShell 运行（使用原实验 Python 和已有模型，无须重新训练先验）：

```powershell
cd D:\desktop\code
.\run_ca_objective_logging.ps1 check
.\run_ca_objective_logging.ps1 run
```

默认仅预检；只有 `run` 才启动正式实验。`test` 可执行新增日志测试。
需要其他 Python 时，追加 `-PythonExe "完整的python.exe路径"`。

## 实验范围和输出

新配置完整沿用原 CA v9 参数：U=5/7/9、各10次重复、种群100、40个代次记录（含初始代）、每架无人机4000次目标函数评估，共210个单无人机规划任务。
此次修改不增加代数或评估预算，不更改候选生成、修复、筛选、随机种子或停止规则。
4000次评估通常对应40个记录点，前2000次约20点，不是2000个训练episode。

新结果目录：`D:\desktop\code\runs\formal_experiment_ca_v9_objective_logs_v1`。
日志位置：`ca_diffusion_es\N5\trial_00\history_drone0.csv`（其他规模、重复和无人机类似）。
原正式实验结果目录不变。原正式运行器不支持断点续跑，会拒绝向非空结果目录重新运行；如需再次从头运行，请将新JSON的 `output_dir` 改为另一个新目录，不要删除旧实验。
新增命令不运行其他基线算法，也不重新训练先验。

## 新增字段

`offspring_postrepair_*`：本代完成重采样/可行性修复后的子代，尚未与父代进行生存筛选。
`population_selected_*`：父代和子代合并并筛选后的种群。
初始代尚无子代，其count为0、统计量为NaN，绘图应跳过，不能填成零。

每个阶段都有 `all`（全部候选）和 `feasible`（约束违反量 <= 1e-9）两个子集。
各自保存 count，以及以下指标的 finite_count、mean、std、min、median、max：

- `length_m`：路径长度，单位m。
- `communication`：算法原有通信代价，无量纲；不是dB路径损耗。
- `fixed_score`：固定尺度、固定权重的描述性综合代价。
- `constraint_violation`：算法原有约束违反量。

优先查看的曲线字段：

- `offspring_postrepair_all_length_m_mean`
- `offspring_postrepair_all_communication_mean`
- `offspring_postrepair_all_fixed_score_mean`

综合代价 = `(1-w) * length / L_ref + w * communication / C_ref`。
本配置 `w=0.65, L_ref=2000, C_ref=1.05`。每行也记录这三个参数，避免随种群变化重新归一化。
这个分数仅用于诊断，没有约束罚项，不参与搜索或替代原来的多目标排序，也不能称为算法实际优化的唯一目标函数。

std为本代候选个体之间的样本标准差（ddof=1），不是重复实验之间的标准差或均值置信区间。
单个有效值的std记0；空子集或全非有限值记NaN。finite_count用于说明实际参与某项统计的数量。
对10次实验汇总时，应另算运行之间的标准差，不能直接平均本字段充当该误差带。

## 记录阶段的限制

修复前只有几何可行性检查，没有完整的目标函数评估。因此保留原有的 `raw_offspring_feasible_ratio` 等诊断，而不额外评估未修复路径以免改变预算。
新的子代目标值是修复后、筛选前的统计；不是修复前的原始提案，也不是逐候选解的完整明细。
HV、indicator和原始可行比例继续原样记录。新目标均值允许上下变化，不做累计最优或时间平滑，也不保证一定出现强烈波动或单调收敛。

日志导出已改为保留所有出现过的字段，避免新增字段被原有固定表头静默丢弃。
旧报告依旧画HV；新实验结束后需用上述新字段绘制目标变化曲线。
