# AF3 主后端、统一多环境镜像、双后端验证与 ESM 评分实施方案

## 总体设计

- AF3 固定为主 fold backend，对全部候选运行。
- 第二 backend 每次选择 none、protenix-v2 或 OpenDDE 通用模型；ABAG 仅为显式 checkpoint 覆盖。
- AF3 ipTM 大于等于 0.70 且输入与 target feature 有效的候选进入第二 backend；不要求 AF3 结构成功。
- Protenix/OpenDDE 复用 AF3 target-only 的 MSA 和 template；禁止静默重搜或联网下载。
- AF3 template 无法转换时，第二 backend 降级为 MSA-only。
- ESMFold 对全部 Binder 运行；ESM-IF 对具有有效 AF3 复合物的 Binder B 链运行。
- 四套模型环境和本地 MSA/template 工具链安装在同一个 Docker 镜像中，每个 stage 使用独立容器和最小目录挂载。
- 共识、无监督离群和 ESM 指标仅用于注释、排序和人工审阅，不硬过滤。
- 双后端共识完成后，用 AF3 结构对最终候选池执行三层聚类。

## 统一 Runtime 镜像

从固定 digest 的 nvidia/cuda:12.6.3-base-ubuntu24.04 通过 Dockerfile 构建：

    /opt/envs/af3        Python 3.12 + JAX/XLA
    /opt/envs/opendde    Python 3.11 + PyTorch 2.7.1 cu126
    /opt/envs/protenix   Python 3.12 + PyTorch 2.7.1 cu126
    /opt/envs/esm        Python 3.9 + PyTorch 1.12.1 cu112

源码和依赖基线：

- AF3：/home/structure/Software/alphafold3-3.0.3 及其 uv.lock。
- OpenDDE：commit 266ce4c，以本机 uv 环境冻结依赖。
- Protenix：2.0.0，以本机 Conda 环境为依赖基线。
- ESM：fair-esm 2.0.1、OpenFold 2.2.0、Deepspeed 0.19.0。
- MSA/template：GPU-enabled MMseqs2 commit 8cc5ce、patched HMMER 3.4、Kalign 和 ColabFold 1.6.1。

镜像提供 fold-runtime af3、protenix、opendde、esm-if、esmfold、prepare-features 子命令。MMseqs2 对 padded 数据库默认使用调度器选择的一张空闲 GPU 和 --gpu 1；GPU 不可用时明确失败，不静默回退 CPU。每个命令使用绝对解释器路径，并重建独立的 PATH、PYTHONPATH、LD_LIBRARY_PATH、HOME、CUDA_VISIBLE_DEVICES 和 TORCH_HOME。

宿主机 Aerith 负责调度；每个 stage 从同一镜像启动独立容器。模型、数据库、MSA/template 均只读挂载，仅输出目录可写，并使用 network none。

## AF3 Feature 复用

缓存布局：

    work/features/<target-sha256>/af3/
    ├── target_data.json
    ├── unpaired.a3m
    ├── paired.a3m
    ├── templates/
    ├── secondary_adapter/
    │   ├── templates.a3m
    │   └── mmcif/
    └── manifest.json

Target A 链直接复用 AF3 paired/unpaired MSA。Binder B 链仅使用 query-only unpaired MSA。

AF3 template adapter 解析排名前四个 template 的 mmCIF、PDB ID、author chain、queryIndices 和 templateIndices，生成 Protenix/OpenDDE 可用的 templates.a3m 与按 PDB ID staging 的 mmCIF。若没有有效 template，secondary 显式关闭 template 并使用 MSA-only，绝不重新搜索。

## Pipeline 顺序

1. AF3 target-only feature。
2. AF3 全量复合物预测。
3. ESMFold 全量 Binder 单体预测。
4. 对有效 AF3 复合物运行 ESM-IF。
5. AF3 Biotite 和 Rosetta。
6. 选择 AF3 ipTM 大于等于 0.70 的候选。
7. 转换 AF3 template contract。
8. Protenix-v2 或 OpenDDE 通用模型预测；OpenDDE-ABAG 作为显式可选 checkpoint。
9. 第二后端 Biotite 和 Rosetta。
10. 双后端表位、pose、fold 共识。
11. 稳健 cohort 离群检测。
12. 形成最终聚类候选池。
13. 使用 AF3 结构做三层聚类。
14. 输出 shortlist 和人工审阅名单。

GPU stage 按 AF3、ESM tools、secondary backend 顺序串行独占 GPU 池，每个 stage 内一张 GPU 一个容器。

## Gate、共识与候选池

Secondary gate 仅要求 AF3 ipTM 大于等于 0.70、AF3 target feature 有效和输入序列有效。AF3 CIF、链、几何或表位失败不阻止 secondary 复核。

双后端聚类候选池要求 secondary 成功，并且 AF3 或 secondary 任一后端通过 geometry/epitope design pass。跨后端不一致和 anomaly 不排除候选。

表位共识输出 target contact Jaccard、interface pair Jaccard 和两边 epitope coverage。Purity 仅保留在阶段详细表中，不过滤、不排序，也不进入公开 CSV。

Pose 共识先对共同 target C-alpha 做稳健 Kabsch 核心对齐，再输出 target 全局/核心 RMSD、target-aligned Binder interface/all-chain RMSD、Binder center displacement 和 interface lDDT。

Fold 共识允许 Binder 独立最佳叠合，输出 Binder TM-score 和 RMSD；该 TM-score 不作为 binding pose 指标。

## 无监督离群

- AF3+Protenix 与 AF3+OpenDDE 分开建立 cohort。
- 少于 30 个双成功样本不生成 outlier。
- 使用 median/MAD robust deviation，不设置固定 contamination。
- 至少两个表位/pose 指标 robust deviation 超过 3.5，或两边均有至少 5 个 target 接触且 Jaccard 小于 0.10，才触发人工审阅。
- anomaly 不影响 final_pass，也不重复参与排序。

## ESM

ESMFold 对全部 Binder 输出 pLDDT 和单体结构。对存在 AF3 B 链的任务，计算允许 Binder 独立叠合的 ESMFold-vs-AF3 TM-score 和 RMSD。

ESM-IF 使用 AF3 最佳复合物中的 Binder B 链，输出 log-likelihood 和 perplexity。ESM 指标只用于注释和排序。

## 排序、聚类和输出

排序依次考虑：任一后端 design pass、参考表位 best coverage、双后端成功、contact/pair Jaccard、interface lDDT、target-aligned Binder interface RMSD、较差 interface PAE、较低 ipTM、Rosetta、ESM-IF、ESMFold、job ID。

排序后使用候选池的 AF3 结构执行 Binder fold、complex pose 和 target epitope fingerprint 聚类。

运行根目录只输出同一列顺序的 `all_results.csv`、`candidates.csv` 和 `final_shortlist.csv`，另有 `manifest.json`、`resolved_config.yaml` 与 `stages/`。完整宽表、manual review、命令日志、Rosetta、ESM 和 Foldseek 产物按十个编号阶段写入各自的 `tables/`、`logs/` 和 `artifacts/`。公开接触残基使用带链号的 1-based 输入序列位置。

`stages/09_consensus/tables/manual_review.csv` 汇总跨后端表位/pose 离群、AF3 异常但 secondary 成功、target 无法稳定对齐、相同 fold 不同 pose，以及 secondary 挽救 AF3 的候选。

## 失败语义与验收

启用的 stage 出现未预期失败时保留成功结果、manifest 标记 partial，并默认非零退出；allow_partial=true 才允许零退出。低于 gate、MSA-only 降级和 cohort 样本不足属于正常状态。

验收包括四环境 import/CLI/GPU smoke、stage 最小挂载、ESM CUDA 隔离、AF3 feature 逐字节复用、template adapter、gate 边界、AF3 异常的 secondary 复核、稳健对齐、离群检测、ESM 全量、OpenDDE 通用 checkpoint、ABAG checkpoint 显式覆盖、checkpoint 缓存隔离、双后端 Rosetta，以及 AF3 到 consensus 和 clustering 的单样本禁网集成测试。
