# Aerith `final_shortlist.csv` 字段说明

本文说明 Aerith 输出 schema v2 中 `final_shortlist.csv` 的 83 个表头。`all_results.csv`、`candidates.csv` 和 `final_shortlist.csv` 使用完全相同的列；区别只是包含的行不同：

- `all_results.csv`：所有输入 job；
- `candidates.csv`：`candidate_pass=true` 的 job；
- `final_shortlist.csv`：三层聚类后，每个 diversity cell 选出的一个质量代表。

实际阈值必须以同一 run 目录中的 `resolved_config.yaml` 为准。本文提到的 5 Å、5 对接触、coverage 0.30 和 AF3 ipTM 0.70 是当前 `balanced` 生产默认值。

## 方向图例

| 标记 | 含义 |
|---|---|
| `↑` | 通常越大越好 |
| `↓` | 通常越小越好 |
| `↓，更负更好` | 数值越低越有利；负值优于正值 |
| `门槛型` | 达到门槛很重要，超过门槛后不应简单认为越大越好 |
| `情境依赖` | 不能单独作单调排序，必须结合其他指标和结构检查 |
| `—` | ID、路径、状态或分类字段，没有数值方向 |

空白值表示指标不可用、未运行或不适用，绝不等于 0。比较数值前应先检查对应的 `*_status`。

## 筛选逻辑先读

### Backend pass

`primary_pass` 和 `secondary_pass` 分别按各自预测结构计算：

```text
backend_pass = interface_status == success
               AND interface_contact_pair_count >= 5
               AND（未配置参考表位 OR epitope_coverage >= 0.30）
```

接触定义为 target/binder 任意聚合物重原子距离不超过 `interface.distance`，默认 5.0 Å。当前版本不使用 epitope purity 做硬过滤；purity 只存在于内部分析表中作为注释。

### Secondary gate 与 candidate pass

主后端固定为 AF3。启用 Protenix/OpenDDE 次后端时：

```text
secondary_gate_pass = AF3 fingerprint 有效且 primary_iptm >= 0.70

candidate_pass = secondary_gate_pass
                 AND secondary_status == success
                 AND (primary_pass OR secondary_pass)
```

因此，`candidate_pass=true` 不表示两个后端都必须 pass。它表示样本进入了次后端交叉验证、次后端成功完成，并且至少一个后端通过几何/coverage。后端之间的表位或 pose 分歧由 consensus 字段和 `manual_review` 标记。未启用次后端时，`candidate_pass` 等于 `primary_pass`。

Rosetta、ESM 和 consensus 指标默认参与注释或排序，但不作为硬过滤阈值。

## 1. 标识、输入与链定义

| 字段 | 方向 | 类型/单位 | 说明 |
|---|---:|---|---|
| `job_id` | — | 字符串 | sanitize 后的唯一 job 主键，也是输出目录和缓存 manifest 的键。 |
| `sample_no` | — | 字符串/整数样式 | 原始输入 CSV 的样本编号；保留原值，不作为质量排序依据。 |
| `run_name` | — | 字符串 | 原始输入 CSV 的设计名或运行名。 |
| `source_row_number` | — | 1-based 行号 | 该记录在原始 CSV 中的物理行号，包含表头后的真实行号，用于追溯并防止行错位。 |
| `target_chain` | — | 单字符链 ID | target 链；当前默认 `A`。 |
| `binder_chain` | — | 单字符链 ID | binder 链；当前默认 `B`，必须与 target chain 不同。 |
| `target_sequence` | — | 氨基酸序列 | 本次 job 使用的 target 序列。单次 run 强制所有行共享同一 target sequence。 |
| `binder_sequence` | — | 氨基酸序列 | 设计 binder 序列。 |
| `configured_epitope_residues` | — | 链限定残基列表 | 用户配置的参考表位，例如 `A:405;A:409;A:436`。编号是 target 序列的 1-based 位置，不是原始 PDB author residue number。未配置时为空。 |

## 2. 筛选、门控与人工复核

| 字段 | 方向 | 类型 | 说明 |
|---|---:|---|---|
| `primary_pass` | `true` 较好 | 布尔 | AF3 结构是否通过 backend pass：界面解析成功、接触对达到门槛，并在配置表位时达到 coverage 门槛。 |
| `secondary_gate_pass` | 门槛型 | 布尔 | 是否因 AF3 ipTM 达到次后端门槛而实际进入 Protenix/OpenDDE；不是独立质量分数。 |
| `secondary_pass` | `true` 较好 | 布尔 | 次后端结构是否通过相同的界面几何/coverage 规则。未进入次后端时通常为 false/空。 |
| `candidate_pass` | `true` 为候选 | 布尔 | 是否进入 candidates 和聚类候选池。启用次后端时使用上文交叉验证逻辑。 |
| `selection_reasons` | — | 分号分隔标签 | 解释入选或未入选，例如 `primary_pass`、`secondary_pass`、`secondary_rescue`、`secondary_not_gated`、`secondary_failed`、`primary_geometry_failed`、`primary_coverage_failed`。 |
| `manual_review` | `false` 更直接 | 布尔 | `true` 表示需要人工看结构/日志，并不等于自动淘汰。它可用于保留有价值但后端分歧明显的设计。 |
| `manual_review_reason` | — | 分号分隔标签 | 常见值包括 `different_epitope`、`different_binder_fold`、`different_pose`、`same_fold_different_pose`、`robust_multimetric_anomaly`、`secondary_rescue`、`consensus_or_target_alignment_failure`。 |

## 3. Primary 与 secondary backend 指标

以下每行同时解释 primary 和 secondary 两列。Primary 当前是 AF3；secondary 是用户选择的 Protenix 或 OpenDDE。不同 backend 的原生分数校准不同，因此 `ranking_score`、pLDDT 等更适合在同一 backend 内比较；跨 backend 一致性应优先看 consensus 指标。

| Primary 字段 | Secondary 字段 | 方向 | 范围/单位 | 说明 |
|---|---|---:|---|---|
| `primary_backend` | `secondary_backend` | — | 名称 | 实际预测后端，例如 `alphafold3`、`opendde`、`protenix`、`none`。 |
| `primary_status` | `secondary_status` | 成功优先 | 状态 | 预测 adapter 状态。常见值为 `success`、`missing`、`error`；secondary 还可为 `not_selected` 或 `disabled`。`not_selected` 是 gate 未命中，不是推理失败。 |
| `primary_ranking_score` | `secondary_ranking_score` | `↑` | backend 原生分数 | 后端用于选择最佳 sample/model 的 ranking score。越高通常越好，但不同 backend 的绝对值不可直接等价。 |
| `primary_iptm` | `secondary_iptm` | `↑` | 通常 0–1 | 复合物链间相对构象置信度。越高通常表示界面/装配更可信；不是实验结合亲和力。Primary ipTM 还用于 secondary gate。 |
| `primary_ptm` | `secondary_ptm` | `↑` | 通常 0–1 | 整体拓扑置信度。对 binder screening，通常低于 ipTM、界面 PAE 和实际接触几何的优先级。 |
| `primary_plddt_global_mean` | `secondary_plddt_global_mean` | `↑` | 0–100 | 整个预测复合物的平均 pLDDT。Aerith 已统一为 0–100 尺度；它是局部结构置信度，不直接表示结合强度。 |
| `primary_best_model_path` | `secondary_best_model_path` | — | 文件路径 | 该 backend 选中的最佳 mmCIF。通常为绝对路径；移动 run 目录后应同步检查路径。 |
| `primary_interface_status` | `secondary_interface_status` | 成功优先 | 状态 | Biotite 是否成功解析 A/B 标准蛋白重原子并计算界面。Rosetta 失败不会把该几何状态改成失败。 |
| `primary_interface_contact_pair_count` | `secondary_interface_contact_pair_count` | 门槛型 | 残基对数量 | 具有至少一对重原子距离 ≤5 Å 的唯一 target–binder 残基对数。默认至少 5 对才通过；更多接触常意味着更大界面，但不是无限越多越好。 |
| `primary_target_interface_residues` | `secondary_target_interface_residues` | — | 链限定列表 | 与 binder 接触的 target 序列位置，例如 `A:405;A:409`。 |
| `primary_binder_interface_residues` | `secondary_binder_interface_residues` | — | 链限定列表 | 与 target 接触的 binder 序列位置，例如 `B:15;B:18`。 |
| `primary_interface_residue_pairs` | `secondary_interface_residue_pairs` | — | 链限定残基对 | 所有唯一接触对，例如 `A:405-B:15;A:409-B:18`。可用于复核具体 pose，而不只是接触计数。 |
| `primary_epitope_overlap_residues` | `secondary_epitope_overlap_residues` | `↑` | 链限定列表 | target 接触集合与配置表位的交集。没有配置参考表位时为空。 |
| `primary_epitope_overlap_count` | `secondary_epitope_overlap_count` | `↑` | 残基数 | 命中的配置表位残基数量。跨不同表位大小比较时应使用 coverage，而不是只比较 count。 |
| `primary_epitope_coverage` | `secondary_epitope_coverage` | `↑` | 0–1 | `命中表位残基数 / 配置表位残基总数`。例如配置 3 个表位、命中 1 个时为 0.3333。当前默认硬门槛 0.30；未配置表位时为空且不启用该门槛。 |
| `primary_interface_pae_mean` | `secondary_interface_pae_mean` | `↓` | Å | target→binder 与 binder→target 的界面 PAE 均值。越低说明界面相对位置越可信。只在实际界面残基上计算，空值不是 0。 |
| `primary_biotite_bsa_total` | `secondary_biotite_bsa_total` | 情境依赖，通常较大更充分 | Å² | Biotite 计算的双侧总 buried surface area：`SASA_target + SASA_binder - SASA_complex`。较大通常表示界面更广，但受 binder 大小影响，过大也可能来自错误贴合，不能单独作为越大越好的硬指标。它是双侧总值，不应直接与采用单侧/除以 2 定义的文献值混用。 |
| `primary_rosetta_status` | `secondary_rosetta_status` | 成功优先 | 状态 | Rosetta InterfaceAnalyzer 状态，常见为 `success`、`error`、`timeout`、`skipped`、`disabled`。Rosetta 失败时 Biotite 几何仍可成功。 |
| `primary_rosetta_dG_separated_per_dSASA_x100` | `secondary_rosetta_dG_separated_per_dSASA_x100` | `↓，更负更好` | Rosetta energy unit/Å² ×100 | `dG_separated / dSASA_int × 100`。越低越有利，负值优于正值；正值越大不是越好。它不是 kcal/mol，也不是实验 ΔG。当前使用 ref2015、原始预测 pose、`pack_input=false`、`pack_separated=true`，应主要在同一 run/协议内相对比较。 |
| `primary_rosetta_packstat` | `secondary_rosetta_packstat` | `↑` | 通常 0–1 | 界面 packing/空腔填充质量，越高通常越紧密。Aerith 默认固定 Rosetta 随机种子保证恢复运行可复现。 |

## 4. ESM 指标

| 字段 | 方向 | 范围/单位 | 说明 |
|---|---:|---|---|
| `esmfold_status` | 成功优先 | 状态 | Binder 单链 ESMFold 是否成功。它不预测 target–binder 结合。 |
| `esmfold_plddt` | `↑` | 0–100 | Binder 单链 ESMFold 平均 pLDDT，反映序列自身折叠置信度。高值不保证目标结合正确。 |
| `esmfold_af3_binder_tm` | `↑` | 0–1 | 将 ESMFold binder 与 AF3 binder 链独立叠合后的 TM-like fold agreement。越高表示 binder fold 在两种模型中更一致；独立叠合消除了复合物 pose，因此该列不评价结合方位。 |
| `esm_if_status` | 成功优先 | 状态 | ESM inverse folding 对 AF3 binder backbone 的序列评分是否成功。 |
| `esm_if_log_likelihood` | `↑`，即越不负越好 | 每残基平均 log-likelihood | 给定 AF3 binder backbone 时，ESM-IF 对设计序列的兼容性。例：`-0.8` 优于 `-1.5`。它评价 sequence–backbone compatibility，不是结合亲和力。 |
| `esm_if_perplexity` | `↓` | 无量纲，通常 ≥1 | `exp(-esm_if_log_likelihood)`；越低越好。它与 log-likelihood 是同一信息的单调变换，不应在人工加权时重复计数。 |

## 5. Primary/secondary 共识指标

所有固定坐标系指标都先把 secondary target robustly 对齐到 primary target，再观察 binder；因此它们测量的是两个后端对 pose 的一致性，而不是单个后端的绝对正确性。

| 字段 | 方向 | 范围/单位 | 说明 |
|---|---:|---|---|
| `consensus_status` | 成功优先 | 状态 | 常见为 `success`、`not_available`、`not_applicable` 或 `error`。未进入 secondary 的行通常是 `not_available`。 |
| `consensus_target_alignment_rmsd` | `↓` | Å | robust target 对齐后保留 target 残基的 RMSD。越低表示两个后端对共享 target 的构象更一致；异常大值可能说明 target 对齐/构象本身有问题。 |
| `consensus_binder_fixed_frame_rmsd` | `↓` | Å | 对齐 target 后、不再单独对齐 binder 时的全 binder RMSD，同时包含 fold 与结合 pose 差异。越低表示整体复合物方位更一致。 |
| `consensus_interface_fixed_frame_rmsd` | `↓` | Å | 对齐 target 后，两个后端共同 binder 界面残基的 RMSD。越低表示界面局部 pose 更一致。 |
| `consensus_interface_lddt` | `↑` | 0–1 | 基于两个后端共同界面接触区域内距离变化的 interface lDDT-like 一致性。越高越一致。 |
| `consensus_binder_fold_tm` | `↑` | 0–1 | 将两个 backend 的 binder 单独叠合后的 TM-like fold agreement。高值表示 fold 相同，即使复合物 pose 可能不同。 |
| `consensus_epitope_jaccard` | `↑` | 0–1 | 两个后端 target 接触残基集合的 Jaccard：交集/并集。1 表示命中相同 target 区域，0 表示完全不重叠。 |
| `consensus_different_pose` | `false` 通常更一致 | 布尔 | 默认在 binder fold 相同（TM ≥0.50）但 target 固定坐标系下 binder RMSD >5 Å 时为 true。它触发人工复核，不做默认硬淘汰；有时代表值得保留的替代 pose。 |

## 6. 三层聚类与代表字段

聚类只对 `candidate_pass=true` 的候选池运行。当前 balanced 默认大致为：binder fold TM 0.50/coverage 0.80，complex multimer TM 0.65、chain TM 0.50、interface lDDT 0.65，epitope contact-set Jaccard 0.50。最终以 `resolved_config.yaml` 为准。

| 字段 | 方向 | 类型 | 说明 |
|---|---:|---|---|
| `clustering_status` | 成功优先 | 状态 | 三层聚类阶段状态。若 Foldseek 部分失败，可能为 partial/error；应结合 stage 日志判断。 |
| `binder_cluster_id` | — | 分类 ID | Binder 单链结构的 Foldseek fold cluster。相同 ID 表示 binder fold 相似，不表示 pose 或表位相同。 |
| `binder_cluster_size` | — | 候选数 | 该 binder fold cluster 中的 candidate 数量。大 cluster 只表示该 fold 常见，不代表质量更高。 |
| `is_binder_quality_representative` | `true` 为该层代表 | 布尔 | 是否为该 binder cluster 按质量规则选出的代表。false 不表示不合格。 |
| `complex_cluster_id` | — | 分类 ID | 完整 target–binder 复合物的 Foldseek multimer/pose cluster。相同 ID 表示整体装配相似。共同 target 可能影响聚类，因此还需结合 epitope 和 consensus。 |
| `complex_cluster_size` | — | 候选数 | 该 complex pose cluster 的 candidate 数量；不是质量分数。 |
| `is_complex_quality_representative` | `true` 为该层代表 | 布尔 | 是否为该 complex cluster 的质量代表。 |
| `epitope_cluster_id` | — | 分类 ID | 按 target 接触残基集合做确定性 greedy Jaccard 聚类。不同 binder fold 只要命中相似表位，也可进入同一 epitope cluster。 |
| `epitope_cluster_size` | — | 候选数 | 该 epitope cluster 的 candidate 数量；表示表位模式常见程度，不代表质量。 |
| `is_epitope_quality_representative` | `true` 为该层代表 | 布尔 | 是否为该 epitope cluster 的质量代表。 |
| `diversity_cell_id` | — | 组合分类 ID | `binder_cluster_id|complex_cluster_id|epitope_cluster_id`。它同时编码 fold、pose、epitope 三个维度。 |
| `is_final_representative` | `true` 为最终代表 | 布尔 | 是否为其三层 diversity cell 选出的最终代表。在 `final_shortlist.csv` 中应全部为 true。 |
| `final_rank` | `↓` | 正整数 | 最终代表的词典序质量排名，1 最优。它不是线性综合分数，rank 1 与 rank 2 的差距没有定量意义，也不能跨不同配置/run 直接比较。 |

## 最终代表与 final rank 的真实排序规则

### 每个 diversity cell 选谁

同一个 `(binder cluster, complex cluster, epitope cluster)` cell 内只保留一个 candidate。当前 cell 内质量键依次为：

1. `candidate_pass=true`；
2. primary epitope coverage 越高；
3. primary interface PAE 越低；
4. primary Rosetta normalized dG 越低、越负；
5. primary packstat 越高；
6. primary ipTM 越高；
7. `job_id` 作为确定性 tie-breaker。

`is_binder_quality_representative`、`is_complex_quality_representative` 和 `is_epitope_quality_representative` 是各自单层 cluster 的代表标记；它们与三层 cell 的 `is_final_representative` 不是同一个概念。

### final_rank 如何排列 cell representatives

选出每个 cell 的代表后，`final_rank` 按以下优先级依次比较：

1. candidate pass；
2. 两个 backend 中较高的 epitope coverage：越高；
3. `consensus_epitope_jaccard`：越高；
4. 内部字段 `consensus_interface_pair_jaccard`：越高；
5. `consensus_interface_lddt`：越高；
6. `consensus_interface_fixed_frame_rmsd`：越低；
7. primary/secondary 中较差的 interface PAE，即两者最大值：越低；
8. primary/secondary 中较差的 Rosetta normalized dG，即两者最大值：越低、越负；
9. primary/secondary 中较差的 packstat，即两者最小值：越高；
10. primary/secondary 中较差的 ipTM，即两者最小值：越高；
11. ESM-IF log-likelihood：越高、越不负；
12. ESMFold pLDDT：越高；
13. primary backend ranking score：越高；
14. `job_id` 确定性 tie-breaker。

`consensus_interface_pair_jaccard` 参与 rank，但不在公共 83 列中；如需完整审计，可查看 `stages/09_consensus/tables/consensus_results.csv`。缺失的排序指标按最差值处理，而不是按 0 处理。

## 推荐审阅顺序

1. 先确认 `candidate_pass=true`、关键 `*_status=success`，并阅读 `manual_review_reason`。
2. 有参考表位时先看 primary/secondary coverage 和 `consensus_epitope_jaccard`。
3. 看 interface PAE、ipTM、接触对和链限定接触残基，确认两个 backend 都形成合理界面。
4. Rosetta dG 选择更低、更负者，packstat 选择更高者；不要把正 dG 解释成越大越好。
5. 用 ESMFold/ESM-IF 排除明显不稳定或 sequence–backbone 不兼容的 binder。
6. 用 consensus RMSD/lDDT/TM 判断是 fold 分歧、pose 分歧还是表位分歧。
7. 最后打开 primary/secondary mmCIF 做结构人工检查，再决定实验优先级。

## 必须保留的解释边界

- 所有模型分数都是计算指标，不等于实验结合常数、表达量、溶解性、聚集性或免疫原性。
- pLDDT/pTM/ipTM 是置信度，不是自由能；Rosetta normalized dG 也不是 kcal/mol。
- Contact count 和 BSA 都受 binder 大小影响，不能脱离 coverage、PAE、packing 和结构检查单独排序。
- Primary/secondary 原生分数可能有不同校准。跨后端判断应优先使用 consensus，而不是直接比较两个 ranking score 的绝对大小。
- `manual_review=true` 是风险/分歧提示，不是自动失败；`secondary_rescue` 尤其需要同时查看两个结构。
- 所有残基编号都是输入序列的 1-based position，并显式带链号；更新 target 序列后必须重新映射参考表位。
- 修改 backend、序列、模型、镜像、ESM 配置或 feature fingerprint 会影响缓存/run identity；修改筛选阈值后应保留新的 `resolved_config.yaml` 以便审计。
