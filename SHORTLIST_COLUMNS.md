# Aerith schema v3 决策表与多后端审阅表

Aerith schema v3 将“做决策所需的数据”和“多后端完整审计数据”分开输出。

三份决策 CSV 使用完全相同的 55 列 schema：

- `all_results.csv`：全部输入 job；
- `candidates.csv`：`candidate_pass=true` 的 job；
- `final_shortlist.csv`：三层 diversity cell 的最终代表。

`backend_review.csv` 包含全部输入 job 和 108 列：前 55 列与决策表相同，后 53 列保留
primary、secondary、详细 consensus 和分后端 ESMFold 比较值。需要联合审阅两个 backend
时应以该文件为准。

空白值表示指标不可用、未运行或不适用，绝不等于 0。残基编号是输入序列的 1-based
position；公开残基字段始终带链号，例如 `A:405`、`B:15`、`A:405-B:15`。

## Effective backend 语义

Effective backend 是该 job 唯一用于 ESM-IF、Foldseek、表位聚类和单结构质量排序的结构。
一个 backend 只有在最佳模型存在且界面几何解析成功时才有资格成为 effective。两个
backend 都有资格时，按以下确定性顺序选择：

1. backend pass 为 true；
2. epitope coverage 越高；
3. interface PAE 越低；
4. Rosetta normalized dG 越低、越负；
5. Rosetta packstat 越高；
6. ipTM 越高；
7. 完全相同时选择 secondary。

缺失数值排在有效数值之后。没有合格结构时，job 仍保留在 `all_results.csv`，但
`effective_*` 指标为空。

## 决策表的 55 列

### 身份与输入（9 列）

| 字段 | 说明 |
|---|---|
| `job_id` | sanitize 后的唯一 job 主键。 |
| `sample_no` | 输入 CSV 的样本编号。 |
| `run_name` | 输入设计名或运行名。 |
| `source_row_number` | 输入 CSV 中的物理行号，用于追溯。 |
| `target_chain` | target 链 ID。 |
| `binder_chain` | binder 链 ID，必须与 target chain 不同。 |
| `target_sequence` | 本次 job 使用的 target 序列。 |
| `binder_sequence` | 设计 binder 序列。 |
| `configured_epitope_residues` | 链限定的参考表位；未配置时为空。 |

### 筛选与复核摘要（5 列）

| 字段 | 说明 |
|---|---|
| `candidate_pass` | 是否进入候选池。 |
| `selection_reasons` | 分号分隔的入选/未入选原因。 |
| `manual_review` | 是否需要人工检查；它不是自动淘汰。 |
| `manual_review_reason` | 分号分隔的人工复核原因，例如 backend pose、fold 或接触分歧。 |
| `consensus_status` | 跨 backend 共识计算状态；详细数值在 `backend_review.csv`。 |

### Effective backend（22 列）

| 字段 | 方向/说明 |
|---|---|
| `effective_backend` | 被选中的实际 backend 名称。 |
| `effective_selection_reason` | 唯一合格 backend 或质量键中首个决定胜负的字段。 |
| `effective_status` | 被选 backend 的预测状态。 |
| `effective_pass` | 被选 backend 是否通过界面几何和 coverage 规则。 |
| `effective_ranking_score` | ↑，backend 原生模型排名分数。 |
| `effective_iptm` | ↑，链间相对构象置信度，不是结合亲和力。 |
| `effective_ptm` | ↑，整体拓扑置信度。 |
| `effective_plddt_global_mean` | ↑，完整复合物平均 pLDDT，统一为 0–100。 |
| `effective_best_model_path` | effective 最佳模型路径。 |
| `effective_interface_status` | Biotite 界面计算状态。 |
| `effective_interface_contact_pair_count` | 唯一 target–binder 接触残基对数。 |
| `effective_target_interface_residues` | 链限定的 target 接触残基。 |
| `effective_binder_interface_residues` | 链限定的 binder 接触残基。 |
| `effective_interface_residue_pairs` | 链限定的 target–binder 接触对。 |
| `effective_epitope_overlap_residues` | target 接触集合与配置表位的交集。 |
| `effective_epitope_overlap_count` | 命中的配置表位残基数。 |
| `effective_epitope_coverage` | ↑，命中表位残基数 / 配置表位大小。 |
| `effective_interface_pae_mean` | ↓，effective 界面 PAE 均值，单位 Å。 |
| `effective_biotite_bsa_total` | Biotite 双侧总 buried surface area，单位 Å²。 |
| `effective_rosetta_status` | Rosetta InterfaceAnalyzer 状态。 |
| `effective_rosetta_dG_separated_per_dSASA_x100` | ↓、更负更好；不是 kcal/mol。 |
| `effective_rosetta_packstat` | ↑，界面 packing 质量。 |

Rosetta 失败不会清空已成功计算的 Biotite 几何；应先检查
`effective_rosetta_status` 再解释 Rosetta 数值。

### ESM（6 列）

| 字段 | 方向/说明 |
|---|---|
| `esmfold_status` | Binder 单链 ESMFold 状态。 |
| `esmfold_plddt` | ↑，Binder 单链折叠置信度，0–100。 |
| `esmfold_effective_binder_tm` | ↑，ESMFold Binder 与 effective Binder 的 fold agreement。 |
| `esm_if_status` | 对 effective Binder backbone 做 inverse-folding 评分的状态。 |
| `esm_if_log_likelihood` | ↑、越不负越好；sequence–backbone compatibility。 |
| `esm_if_perplexity` | ↓；与 log-likelihood 是同一信息的单调变换。 |

ESMFold 只按 Binder 序列预测一次；它不预测 target–binder 结合。

### 聚类与排名（13 列）

| 字段 | 说明 |
|---|---|
| `clustering_status` | 三层聚类状态。结构提取失败或 Foldseek 漏项时应为 error/partial。 |
| `binder_cluster_id` | effective Binder 的 Foldseek fold cluster。 |
| `binder_cluster_size` | Binder cluster 的候选数。 |
| `is_binder_quality_representative` | 是否为该 Binder cluster 的质量代表。 |
| `complex_cluster_id` | effective A/B complex 的 Foldseek pose cluster。 |
| `complex_cluster_size` | Complex cluster 的候选数。 |
| `is_complex_quality_representative` | 是否为该 Complex cluster 的质量代表。 |
| `epitope_cluster_id` | effective target contact set 的 epitope cluster。 |
| `epitope_cluster_size` | Epitope cluster 的候选数。 |
| `is_epitope_quality_representative` | 是否为该 Epitope cluster 的质量代表。 |
| `diversity_cell_id` | `binder|complex|epitope` 三层组合 ID。 |
| `is_final_representative` | 是否为 diversity cell 的最终代表。 |
| `final_rank` | 最终代表的确定性质量排名，1 最优。 |

只有 Foldseek 输出确认了自成员关系的结构才能成为 singleton。无法进入聚类的 job 不伪造
cluster ID，保留在 all-results/candidates 供复核，但不进入 final shortlist。

## `backend_review.csv` 的额外 53 列

### Backend pass 与 gate（3 列）

- `primary_pass`
- `secondary_gate_pass`
- `secondary_pass`

### Primary/secondary 完整指标（40 列）

每个 prefix（`primary`、`secondary`）各有以下 20 列：

| 后缀 | 说明 |
|---|---|
| `backend` | backend 名称。 |
| `status` | 预测状态。 |
| `ranking_score` | backend 原生排名分数。 |
| `iptm` | ipTM。 |
| `ptm` | pTM。 |
| `plddt_global_mean` | 复合物平均 pLDDT。 |
| `best_model_path` | 最佳模型路径。 |
| `interface_status` | Biotite 界面状态。 |
| `interface_contact_pair_count` | 接触残基对数。 |
| `target_interface_residues` | 链限定 target 接触残基。 |
| `binder_interface_residues` | 链限定 binder 接触残基。 |
| `interface_residue_pairs` | 链限定接触对。 |
| `epitope_overlap_residues` | 配置表位交集。 |
| `epitope_overlap_count` | 配置表位命中数。 |
| `epitope_coverage` | 配置表位 coverage。 |
| `interface_pae_mean` | 界面 PAE 均值。 |
| `biotite_bsa_total` | 双侧总 BSA。 |
| `rosetta_status` | Rosetta 状态。 |
| `rosetta_dG_separated_per_dSASA_x100` | Rosetta normalized dG。 |
| `rosetta_packstat` | Rosetta packstat。 |

例如后缀 `iptm` 对应 `primary_iptm` 和 `secondary_iptm`。不同 backend 的原生分数校准
可能不同，不应把两个 `ranking_score` 的绝对值直接当作同一量表。

### 详细 consensus（8 列）

- `consensus_target_alignment_rmsd`
- `consensus_binder_fixed_frame_rmsd`
- `consensus_interface_fixed_frame_rmsd`
- `consensus_interface_lddt`
- `consensus_binder_fold_tm`
- `consensus_epitope_jaccard`
- `consensus_interface_pair_jaccard`
- `consensus_different_pose`

固定坐标系指标先把 secondary target 对齐到 primary target，再比较 Binder 或界面，因此反映
两个 backend 对结合 pose 的一致性。分歧用于触发人工复核，默认不作为硬过滤。

### 分后端 ESMFold 比较（2 列）

- `esmfold_primary_binder_tm`
- `esmfold_secondary_binder_tm`

两列使用同一个 ESMFold 单链预测，分别与 primary 和 secondary Binder 比较；决策表只保留
与 effective Binder 对应的 `esmfold_effective_binder_tm`。

## 推荐审阅顺序

1. 在三份决策表中检查 `candidate_pass`、`effective_*_status` 和人工复核原因。
2. 有配置表位时查看 effective coverage、接触残基和接触对。
3. 联合解释 interface PAE、ipTM、BSA、Rosetta dG 和 packstat，不单独依赖一个分数。
4. 用 effective ESMFold/ESM-IF 指标检查 Binder fold 与 sequence–backbone compatibility。
5. 用三层 cluster 保留 fold、pose 和 epitope 多样性。
6. 对 `manual_review=true` 或 secondary rescue 的 job，打开 `backend_review.csv` 和两个结构。

所有计算分数都不等于实验结合常数、表达量、溶解性、聚集性或免疫原性。实际阈值以同一
run 的 `resolved_config.yaml` 为准。
