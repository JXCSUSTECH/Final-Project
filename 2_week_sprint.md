## 两周冲刺计划（3 人团队 — A / B / C）

目标：在两周（10 个工作日）内基于现有 DQN baseline，实现并比较两种新的 RL 算法（PPO 与 Actor-Critic），完成可复现实验并提交训练模型与 ≤8 页报告。

参与成员（按请求使用 A/B/C 表示）：
- 成员 A — PTO：负责实现 `agents/cartpole_ppo.py`（PPO 实现与训练）
- 成员 B — PTO：负责实现 `agents/cartpole_actorcritic.py`（Actor-Critic / A2C 实现与训练）
- 成员 C — PTO：负责实验与评估、`train.py` 参数化、结果记录与报告整合

验收准则（两周末必须满足）：
- 两个新算法文件位于 `agents/`：`cartpole_ppo.py`, `cartpole_actorcritic.py`，并能被 `train.py` 调用
- 每个算法完成至少 3 次重复训练（固定种子），并保存模型到 `artifacts/`（*.keras）
- 将所有实验结果（CSV）与学习曲线图放入 `scores/`，并提交 ≤8 页报告（PDF）

高层时间表（每日要点）

Week 1 — 快速原型与接口集成
- Day 1（启动会, 2 小时）
  - 确认目标与分支策略：每人创建 `feature/<name>` 分支并约定 PR 流程。
  - 全员：在本地复现 baseline：`python train.py`（生成 baseline 曲线）。
- Day 2–3（实现骨架）
  - A：实现 PPO 基本网络（policy/value）、采样与更新步骤（可先用单线程、少量 episodes 测试）。
  - B：实现 Actor-Critic 骨架（策略网 + 价值网）、损失与单步训练逻辑。
  - C：参数化 `train.py`（添加 `--agent`、`--episodes`、`--save-path`），统一 score 保存格式（CSV）与绘图接口。
- Day 4（小规模集成测试）
  - 每人提交小 PR，互相 review，运行一次端到端训练（10–30 episodes）验证接口。
- Day 5（周中检视）
  - 汇报进展与问题，决定 Week 2 优先级（超参/稳定性/可视化）。

Week 2 — 超参、稳定性与报告
- Day 6–7（超参试验）
  - A/B：各自进行 2–3 个关键超参试验（例如 `learning_rate`, `gamma`, `entropy_coeff`），每次缩减训练长度以快速验证趋势。
  - C：组织并运行实验（记录 config JSON 到 `scores/`，把结果 CSV 化）。
- Day 8（重复训练与最终模型）
  - 每个算法至少运行 3 次不同 seed 的完整训练，保存模型到 `artifacts/`。
- Day 9（结果整理）
  - 生成学习曲线图、超参敏感性表格；各自写实现与分析段落。
- Day 10（收尾与合并）
  - 全体进行代码审查并合并 PR；生成最终报告 PDF，打包 `artifacts/` 与 `scores/`。

并行与日常实践
- 分支与 PR：`feature/ppo`、`feature/actor-critic`、`feature/experiments`。
- 每日早上：`git pull` 同步（15 分钟），每日结束前 push 并在 PR/Issue 上留进度。
- 遇到冲突：小步提交，及时沟通，优先解决阻塞问题。

开发与运行示例（Windows PowerShell）
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

如果 `train.py` 被参数化后的调用示例：
```
python train.py --agent dqn --episodes 50 --save-path artifacts/dqn.keras
python train.py --agent ppo --episodes 100 --save-path artifacts/ppo.keras
```

建议的任务拆分（可直接在 Issues 中创建）
- Issue 1 — `feature/ppo`：实现 PPO 骨架并添加 smoke test（指派 A）
- Issue 2 — `feature/actor-critic`：实现 Actor-Critic 骨架并添加 smoke test（指派 B）
- Issue 3 — `train.py 参数化`：添加 `--agent`、`--config` 并统一日志格式（指派 C）
- Issue 4 — `实验脚本`：PowerShell / 小批量超参脚本，收集 CSV（指派 C）
- Issue 5 — `报告与图表`：收集图，生成 PDF（指派 C；A/B 提供实验段落）

风险与缓解
- 训练慢或无 GPU：先做小规模（少 episodes）试验，使用固定 seed 做快速比较。
- 算法实现卡住：先实现最小可运行版本（smoke test），待 Week2 做改进。

文档与交付（两周结束时）
- `agents/cartpole_ppo.py`、`agents/cartpole_actorcritic.py`（可运行）
- `artifacts/` 下的最终模型文件（*.keras）
- `scores/` 下的 CSV 和学习曲线图
- 报告 PDF（≤8 页）

---


