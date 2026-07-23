# Aerith maintainer handoff

本文件适用于仓库根目录及全部子目录。它记录必须保留的行为约束、验证方法和清理边界。
用户在后续清理中提供的明确保留/删除清单优先于本文件；未得到明确授权前不得删除项目文件或运行数据。

## 项目定位

Aerith 是蛋白 Binder 筛选的控制面和调度器，不是新的结构预测模型：

- AlphaFold 3 是固定 primary fold backend。
- secondary backend 可选 `none`、`protenix` 或 `opendde`，用于交叉验证。
- AF3、Protenix、OpenDDE、ESMFold、ESM-IF、GPU MMseqs2 和 GPU Foldseek 位于统一的
  `aerith/fold-runtime` Docker 镜像中，但按阶段启动相互隔离的容器。
- Biotite 计算界面几何；Rosetta InterfaceAnalyzer 计算能量和 packing。
- 聚类由 Binder fold、complex pose 和 target-contact epitope 三层组成。
- 当前生产执行模式是单机多 GPU + Docker；Aerith 负责 GPU 分片和阶段依赖。

权威设计与输出说明：

- `README.md`：生产 Quick Start、镜像构建与验收记录。
- `SHORTLIST_COLUMNS.md`：三个 55 列决策 CSV 与 108 列多后端审阅 CSV 的字段语义。

## 不得破坏的行为

1. 单次 run 只能有一个共享 target sequence；target/binder chain 必须非空且不同。
2. CSV 只解析一次并生成不可变 JobSpec；`limit` 必须作用于整个任务计划。
3. job 名 sanitize 后必须唯一；重复 job 和 target mismatch 必须显式失败。完全空白的
   CSV 物理行按解析契约跳过，但 `source_row_number` 必须保留原文件行号，不能发生行错位。
4. 缓存只能在 fingerprint、manifest 和可解析产物全部匹配时命中。
5. CLI 状态文字使用 `cache hit!` 和 `cache missing!`，并显示 stage 与完成数/总数。
6. AF3 target 使用本地 GPU MMseqs2 MSA 和本地 template；Binder 默认 query-only MSA、
   无 paired MSA、无 template。
7. 预测/feature 容器默认 `--network none`，数据库和 checkpoint 只读挂载。
8. 表位编号是 target 输入序列的 1-based position；公开残基字段必须包含链号，
   例如 `A:405`、`B:15`、`A:405-B:15`。
9. 硬表位门槛只使用 coverage；purity 不过滤、不进入公开决策列。
10. Rosetta 失败不得丢弃 Biotite 几何；Rosetta 默认固定随机种子以保证恢复可重复。
11. 缺失指标保持 missing，不得用 0 代填。
12. pipeline 可保留部分产物，但必需阶段失败时必须非零退出，除非 `allow_partial=true`。
13. 三个 run-root CSV 必须同 schema：`all_results.csv`、`candidates.csv`、
    `final_shortlist.csv`（当前 55 列 effective-only schema）；`backend_review.csv`
    单独保存完整 108 列多后端数据。详细日志/表格/产物按 `stages/01_*` 至
    `stages/10_*` 保存。
14. CSV/JSON 采用临时文件和原子 rename；任意非零 subprocess return code（包括负信号）
    都是失败。

## 代码地图

- `src/af3_binder_filter/cli.py`：Typer CLI。
- `src/af3_binder_filter/config.py`、`config_tools.py`、`conf/`：Hydra Structured Config。
- `src/af3_binder_filter/workflow.py`：稳定兼容门面，不放置新的阶段实现。
- `src/af3_binder_filter/orchestration/`：生产编排实现；`PipelineRunner` 显式执行
  十个 stage，`PipelineState` 集中保存跨阶段状态，stage registry 固定名称、顺序和启用条件。
- `jobs.py`、`manifest.py`、`io_utils.py`：任务身份、恢复可靠性和原子 I/O。
- `features.py`、`secondary_features.py`：本地 MSA/template 与后端 feature contract。
- `backends.py`：AF3/Protenix/OpenDDE adapters。
- `interface.py`、`rosetta.py`：界面几何与能量分析。
- `esm_tools.py`、`consensus.py`、`clustering.py`：ESM、跨后端共识和三层聚类。
- `reporting.py`、`output_layout.py`、`progress.py`：公开 CSV、分阶段目录和 CLI 进度。
- `docker/runtime/`：统一 runtime Dockerfile、entrypoint 和依赖 locks。
- `docker/feature-builder/`：GPU MMseqs2/template 预处理 adapter。

## 生产与已退役边界

`esmfold_score.py`、`gpu.py`、`models.py` 和 `af3_json.py` 仍属于生产调用图。
`models.py` 只保留当前 CSV 输入所需的 `BinderCsvRow`。

2026-07-23 已完成引用审计并删除最早版本中不可达的
`pipeline.py`、`esm_score.py`、`ipsae_score.py`、`aggregate.py`、
`af3_runner.py`、`sasa.py` 和 `sequence_metrics.py`。不要恢复旧 Ray/Modin
执行路径；并行计算由当前 process-safe executor 和明确的 GPU shard 负责。

`main.py` 是兼容入口；只有在确认所有用户都使用 `aerith` console script 后才能删除。

工厂只用于真正可替换的运行组件，例如 `InterfaceEnergyEngine`；不要用动态 stage factory
隐藏缓存恢复、manifest 更新、失败传播或进度状态。新增阶段必须先进入固定 registry，再由
`PipelineRunner` 显式编排，并为启用条件和顺序补测试。

## 必须执行的验证

代码修改后至少运行：

```bash
uv run pytest -q
uv run python -m compileall -q src scripts docker
git diff --check
```

配置/容器相关修改还需运行：

```bash
aerith config validate --config <config.yaml>
aerith config doctor --config <config.yaml>
aerith pipeline --config <config.yaml> --dry-run
docker run --rm --gpus all --network none aerith/fold-runtime:local doctor
```

真实 GPU 验收应先使用一条受控样本；不得在普通 PR CI 中自动启动完整生产 screen。

## CI/CD 与 Kubernetes 边界

当前项目不需要 Kubernetes。单台四卡服务器上，Aerith 已经完成 Docker GPU 分片、缓存恢复、
阶段屏障和失败记录；再引入 K8s 会形成双重调度并增加镜像分发、PVC、device plugin、权限和
集群维护成本。

建议当前保持：

```text
GitHub Actions（CPU CI / 镜像发布 / 手动 GPU smoke）
    -> 专用 self-hosted runner（一次独占整台 GPU 主机）
        -> Aerith
            -> GPU-isolated Docker containers
```

只有满足以下条件时才评估 K8s：多物理 GPU 节点、多人共享队列、需要自动扩缩容/故障重调度、
已有长期维护的 Kubernetes 平台，或 Aerith 需要成为常驻 API 服务。进入 K8s 前先抽象
`Executor` 协议，至少支持 `local_docker`，再增加 `slurm` 或 `kubernetes` adapter；不要把
K8s 逻辑直接写死在 workflow。若实验室已有 Slurm，批量科学计算通常优先接 Slurm。

## 清理协议

1. 先生成 tracked、untracked、ignored 和大小清单。
2. 用户给出明确保留清单后，将其作为清理 allowlist。
3. tracked 文件必须逐项说明依赖与删除影响，并使用 `git rm`。
4. ignored 运行产物也不得默认删除；确认后再清理。
5. 清理后运行完整测试、compileall、CLI help 和 `git diff --check`。
6. 在提交前确认工作树中没有模型权重、数据库、真实生产输出、绝对路径配置或密钥。

截至 2026-07-23 的待审阅分类：

- 明确保留候选：`.gitignore`、`.dockerignore`、`pyproject.toml`、`uv.lock`、`src/`、
  `docker/`、`scripts/`、`tests/*.py`、`README.md`、`SHORTLIST_COLUMNS.md`、
  `AGENTS.md`、`LICENSE`。
- 需要用户决定的 tracked 旧文件：`all_seq_PD1_May12.csv`、`candiate.csv`、`main.py`。
- 需要用户决定的本地 ignored 内容：`.venv/`、`.pytest_cache/`、`__pycache__/`、
  `config.yaml`、`results/`、`work/`、`tests/test_hydra_config.py.orig`。
- 外部 `/data/AF3_database`、模型 checkpoint、`/ssd` 生产 screen 和 Docker data-root
  不属于仓库清理范围，除非用户另行明确授权。
