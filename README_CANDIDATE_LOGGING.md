# U=5 逐候选解评估日志

代码使用现有数据、先验模型和CA v9优化器，无须重新训练。

```powershell
cd D:\desktop\code
.\run_ca_candidate_logging.ps1 -Action check
.\run_ca_candidate_logging.ps1 -Action run
```

`check`仅预检，不启动实验；`run`正式运行；`test`执行日志相关测试。
默认使用 `.venv\Scripts\python.exe`，也可指定 `-PythonExe "完整路径"`。

## 范围

- U=5、10次重复，共50个单无人机规划任务。
- 每架每次4000次目标评估，约40个代次（包括初始代），保持原种群100及全部搜索参数。
- 每架每次写入4000条逐候选记录，共约200000条；没有增加目标评估或改变随机种子。
- N5的场景和种子与原正式配置一致，仅缩小到U=5并开启日志。
- 此次为候选诊断运行，不会冒充包含U=5/7/9的完整正式实验。

## 输出

新目录：`D:\desktop\code\runs\ca_v9_candidate_logs_n5_v1`。

例如：`ca_diffusion_es\N5\trial_00\candidates_drone0.csv`。
已有 `history_drone0.csv`（逐代汇总）、Pareto结果、最终路径、records和manifest继续输出。
每完成一架无人机，终端会打印已保存的候选记录条数。
4000行指数据行，不含CSV表头。

旧结果不覆盖。继承的实验运行器拒绝非空输出目录，不支持中断续跑；若重新从头运行，将 `candidate_logging_n5.json` 中的 `output_dir` 改成另一个新目录。

## 逐候选字段

- `method,uav_count,trial,seed,drone_id`：实验身份。
- `evaluation_id`：1至4000连续编号，每行对应一次实际计入预算的目标评估。
- `generation`：所属代数，从0开始；0为初始化。
- `batch_id,index_in_batch,batch_size`：实际评估批次、批内索引（从0开始）、批大小。
- `stage`：`initial_population`、`initial_topology_seed`或`offspring_postrepair_preselection`。
- `length_m,communication_cost`：本次评估的路径长度与通信代价。
- `fixed_cost`：固定尺度描述性分数 `(1-w)*length/Lref + w*communication/Cref`。
- `constraint_violation,is_feasible`：约束违反量及是否可行。
- `min_clearance_m,collision_count,mean_pathloss_db,worst_pathloss_db,separation_violation_m`：对应候选的安全和通信指标。
- `topology_origin`：算法原有来源标记，不等同于“仅在线RRT”标记。
- `length_scale_m,communication_scale,communication_weight`：本配置为2000、1.05、0.65。
- `best_feasible_fixed_cost_so_far`：截至该评估编号的最低可行固定分数。未出现可行解时为NaN。它是诊断指标，不是超体积，也不参与搜索。

## 记录与绘图的准确含义

记录覆盖算法实际计算目标值的全部候选，包括初始种群和额外初始化种子。
子代目标值的评估发生在可行性修复之后、生存筛选之前；未修复提案只经过几何检查，没有计入目标评估，故不额外计算或伪装为这4000条中的目标数据。
后处理选中路径的复核不计为新优化候选。
同一批中的索引仅表示批内存储次序，不表示4000代算法更新。绘制个体原始波动时应标注这一点，优先使用散点并叠加代次均值。
每个候选可能比上一个差，原始代价可以上下变化；历史最低可行代价只会下降或持平。不要用排序、插值或加噪声制造收敛形状。
新日志记录全部值，不做时间平滑、截断或抽样；浮点NaN作为未定义值保存。
日志会略增加记录时间和磁盘开销，因此新旧耗时不应被当作纯算法速度差异。
