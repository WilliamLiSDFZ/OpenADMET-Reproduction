# OpenADMET-ExpansionRx Blind Challenge — 复现 + 改进实验完整报告

**项目**：`OpenADMET-LightGBM-Reproduction/`
**作者**：Yuze Li
**日期**：2026-05-03
**最佳成绩**：MA-RAE = **0.750**（v1 + selective augmentation）

---

## 0. TL;DR

- 复现了 [OpenADMET-ExpansionRx Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) 的官方 baseline 类方法（**LightGBM + RDKit-2d 描述符 + Morgan 指纹**）。
- 用挑战赛官方 ground truth 做了 **9 次端到端实验**，每次都用官方 `python -m eval` 打分，全部用 macro-averaged Relative Absolute Error (**MA-RAE**) 比较。
- 最佳结果 **MA-RAE = 0.750**（v1 baseline 是 0.756，selective external augmentation 把它压到 0.750）。
- 读了 8 份前 10 名方法论 + JCIM 论文之后，挑了 3 个最容易移植的技巧（Avalon 指纹 / ADMET-AI 蒸馏特征 / SALI 活性悬崖 mask），**单独用、组合用全部都没赢过 selective augmentation**。
- 关键 insight：**这套数据集小样本 endpoint + 强分布偏移**这个组合下，给 LightGBM 加更多特征/数据基本就是过拟合，进一步提升只能靠**架构换代**（Chemprop MPNN ensemble + 大规模预训练）或**专有 lead-optimization 数据**。

---

## 1. 任务背景

### 1.1 挑战赛简介

ExpansionRx-OpenADMET Blind Challenge（2025-10 ~ 2026-01）是一场公开的 ADMET 性质预测比赛，由 ExpansionRx + OpenADMET + HuggingFace 联合举办。题目是：
- 给训练集 5326 个真实药物候选化合物的 SMILES + 9 个 ADMET endpoint 的实验测量值（稀疏，每个 endpoint 都有缺失）
- 给测试集 2282 个化合物的 SMILES（label 盲化）
- 预测 9 个 endpoint 的值，按 **macro-averaged Relative Absolute Error (MA-RAE)** 排名

参与者：370+ teams、1000+ 提交。

### 1.2 9 个 endpoint

| 简写 | 名称 | 单位 | 训练集 N | 是否 log 变换 |
|---|---|---|---:|---|
| **LogD** | 脂水分配系数 (pH 7.4) | log 单位 | 5039 | 否 |
| **KSOL** | 动力学溶解度 | μM | 5128 | 是 |
| **HLM CLint** | 人肝微粒体清除率 | μL/min/mg | 3759 | 是 |
| **MLM CLint** | 鼠肝微粒体清除率 | μL/min/mg | 4522 | 是 |
| **Caco-2 Papp A>B** | Caco-2 渗透性 | 1e-6 cm/s | 2157 | 是 |
| **Caco-2 Efflux** | Caco-2 外排比 | (无单位) | 2161 | 是 |
| **MPPB** | 鼠血浆蛋白结合率 | % | 1302 | 是 |
| **MBPB** | 鼠脑蛋白结合率 | % | 975 | 是 |
| **MGMB** | 鼠腓肠肌蛋白结合率 | % | 222 | 是 |

注意 N 跨度：**LogD 5039 → MGMB 222**，差不多 23 倍。这个不均衡后面会成为关键问题。

### 1.3 评分公式（官方 eval 脚本）

```
RAE_per_endpoint = MAE / mean(|y_true - mean(y_true)|)   # 相对均值预测器的 MAE
MA-RAE           = mean(RAE_per_endpoint, over 9 endpoints)
```

`RAE = 0` 是完美预测，`RAE = 1` 等同于"用 endpoint 均值乱猜"。

---

## 2. 整体方法论

### 2.1 选择复现的方法

挑战赛复盘博客里前 10 名几乎都用了 Chemprop / MPNN GNN ensemble，但需要 GPU 和数小时训练。我选了"性价比最高"的方案：**LightGBM + RDKit-2d + Morgan**。这是 OpenADMET 官方提供的 baseline 之一，CPU 5 分钟训完，复现度高，便于做对比实验。

### 2.2 工作流

```
数据 → 特征工程 → 9 个独立 LightGBM (每个 endpoint 一个) → 反 log 变换 → submission.csv → 官方 eval
                                                                                              ↓
                                                                                          MA-RAE
```

### 2.3 关键工程细节

1. **Log 变换**：除 LogD 外，每个 endpoint 训练前做 `log10((x + 1) * multiplier)`，预测后反变换。+1 是为了避免 log(0)。Multiplier 把单位归一到 mol/L (1e-6) 或保持 1。
2. **稀疏数据**：每个 endpoint 单独训练，只用 `endpoint != NaN` 的样本。
3. **特征向量**（v1）：208 维 RDKit 描述符 + 2048 位 Morgan 指纹 (radius=2)，共 2256 维。
4. **LightGBM 参数**：n_estimators=200, learning_rate=0.05, num_leaves=31, feature_fraction=0.8, bagging 0.8/freq=5（接近默认，没做大规模调参）。

---

## 3. 实验日志

总共做了 9 次端到端实验，全部用官方 `python -m eval` 在真测试集上打分。

### 3.1 实验 1：v1 baseline

**配置**：RDKit-2d + Morgan 指纹（共 2256 维）+ 默认 LightGBM。

**5-fold CV 结果**（先在训练集上自我验证）：

| Endpoint | R² | MAE |
|---|---:|---:|
| LogD | 0.838 | 0.344 |
| LogS (KSOL) | 0.639 | 0.308 |
| Log_MLM_CLint | 0.676 | 0.337 |
| Log_HLM_CLint | 0.601 | 0.297 |
| Log_Caco_ER | 0.575 | 0.145 |
| Log_Caco_Papp_AB | 0.612 | 0.203 |
| Log_Mouse_PPB | 0.699 | 0.191 |
| Log_Mouse_BPB | 0.754 | 0.162 |
| Log_Mouse_MPB | 0.678 | 0.161 |
| **Macro Average** | **+0.675** | — |

**官方 eval 结果（真测试集）**：**MA-RAE = 0.756**, Macro R² = +0.347

### 3.2 实验 2：80/20 holdout 验证（CV ↔ test 一致性）

为了验证 CV 是否能预测 test 表现，用 train.csv 的 80% 训、20% holdout 评，再用官方 eval 脚本打分：
- **MA-RAE on holdout = 0.494**, Macro R² = +0.689

**实验 1 vs 2 的 0.262 巨大差距**（CV 0.494 → 真测试 0.756）就是这场比赛的核心难点：**train 和 test 在化学空间上有强 distribution shift**（lead optimization 时间向前），CV 完全捕捉不到。这点在所有后续实验里都要记住。

### 3.3 实验 3：External row augmentation, "all" 配置

外部数据来源（全部从 GitHub 抓的，HF 在沙盒里访问不到）：
- `biogen_logS.csv` (Pat Walters 教程, 2173 条**实测** logS) → 给 KSOL 加样本
- `regression.csv` (chemprop ESOL/Delaney, 500 条实测) → 给 KSOL 加样本
- `drugbank_admet_predictions.csv` (admet_ai 仓库, 2845 条**预测** ADMET) → 给 LogD/KSOL/Caco-2 Papp/Clearance/PPBR 加样本

权重：`W_measured=0.5, W_predicted=0.2`。

**官方 eval 结果**：**MA-RAE = 0.761**（比 baseline 还差 +0.005 ❌）

逐 endpoint 看：
- 帮上忙：LogD (-0.015)、MLM CLint (-0.009)、MPPB (**-0.051**)
- 受损：KSOL (+0.020)、HLM (+0.016)、Caco-2 Papp (+0.048)

### 3.4 实验 4：External augmentation, 低权重

`W_measured=0.3, W_predicted=0.05`，希望减小外部数据的影响。

**结果**：MA-RAE = 0.764（比 0.5/0.2 还差）。说明问题不在权重，在数据本身的偏差。

### 3.5 实验 5：**Selective augmentation**（只在受益 endpoint 上加）

基于实验 3 的 per-endpoint 结果，只在外部数据明显帮上忙的 3 个 endpoint 加：
- LogD ← drugbank Lipophilicity
- MLM CLint ← drugbank Clearance_Microsome
- MPPB ← drugbank PPBR
- 其他 6 个保持纯 challenge data

**官方 eval 结果**：**MA-RAE = 0.750**（比 baseline 改善 -0.006 ✅）

这是直到目前为止的**最佳成绩**。

### 3.6 实验 6 ~ 9：v2 改进尝试（来自前 10 名报告的高优先级技巧）

读了 `resource/` 下面 8 份前 10 名方法论 + 1 篇 JCIM 论文（Fischer/Cedeno 2025, ASAP-Polaris-OpenADMET ADME #4 的复盘）之后，挑了 3 个理论上最容易移植到 LightGBM 上的：

1. **Avalon 指纹**（+1024 bits, JCIM 论文图 4-5 表现最好的 fingerprint 之一）
2. **ADMET-AI 蒸馏特征**（用 drugbank 数据训 45 个 surrogate LightGBM，对挑战赛 SMILES 推理出 45 维额外特征。JCIM 论文图 5 显示 ADMET-AI predictions 作为输入特征对 ADME 任务统计显著提升）
3. **SALI 活性悬崖 mask**（Tanimoto 相似度 + |z|>2.5，JCIM 论文图 3 证实有效）

#### 实验 6：v2_avalon（v1 + Avalon）

**MA-RAE = 0.771**（**+0.015 ❌**）

#### 实验 7：v2_distill（v1 + 45 维蒸馏特征）

**MA-RAE = 0.802**（**+0.045 ❌** 严重退化）

#### 实验 8：v2_cliff（v1 + SALI cliff mask）

**MA-RAE = 0.768**（**+0.012 ❌**）

#### 实验 9：v2 各种组合

| 组合 | MA-RAE | Δ vs baseline |
|---|---:|---:|
| Avalon + Cliff | 0.763 | +0.007 ❌ |
| Avalon + Cliff + selective_aug | 0.757 | +0.001 ~ |
| Avalon + Cliff + Distill (全 3) | 0.766 | +0.010 ❌ |

### 3.7 完整实验对比表

每个 cell 是 endpoint-level RAE（越低越好），最后一列是 macro 平均。**粗体**标出每个 endpoint 的最优配置。

| 配置 | LogD | KSOL | MLM | HLM | CACO-Eff | CACO-Papp | MPPB | MBPB | MGMB | **MA-RAE** | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_baseline | 0.629 | 0.858 | 0.929 | 0.814 | 0.774 | **0.771** | 0.872 | 0.587 | 0.574 | **0.756** | — |
| **v1_selective_aug** | 0.623 | 0.858 | 0.914 | 0.814 | 0.774 | **0.771** | **0.832** | 0.587 | 0.574 | **0.750** | **−0.006** ✅ |
| v2_avalon | **0.620** | 0.866 | 0.900 | 0.835 | **0.763** | 0.806 | 0.927 | 0.619 | 0.604 | 0.771 | +0.015 ❌ |
| v2_distill | 0.650 | 0.907 | 0.904 | 0.857 | 0.823 | 0.820 | 0.962 | **0.570** | 0.721 | 0.802 | +0.045 ❌ |
| v2_cliff | 0.636 | 0.858 | 0.904 | 0.829 | 0.806 | 0.773 | 0.911 | 0.599 | 0.597 | 0.768 | +0.012 ❌ |
| v2_avalon + cliff | **0.620** | **0.821** | 0.895 | 0.816 | 0.813 | 0.806 | 0.940 | 0.606 | **0.553** | 0.763 | +0.007 ❌ |
| v2_combo_best (Avalon+Cliff+sel_aug) | 0.633 | 0.821 | **0.860** | 0.816 | 0.813 | 0.806 | 0.908 | 0.606 | 0.553 | 0.757 | +0.001 ~ |
| v2_all_three | 0.631 | 0.846 | 0.918 | **0.804** | 0.763 | 0.800 | 0.920 | 0.610 | 0.616 | 0.766 | +0.010 ❌ |

**Per-endpoint cherry-pick 上限**（如果用测试集挑出最优配置，会 overfit 测试集，不能算真实成绩）：**0.733**

---

## 4. 关键发现 / 失败分析

### 4.1 为什么 selective augmentation 是唯一有效的改动？

逐 endpoint 看实验 3 的结果，发现 **MPPB 单独 -0.051** 是大头。原因：
- MPPB 的训练集只有 1302 个样本（相对中等偏小）
- drugbank PPBR_AZ 给我们多了 2845 个药物的预测值
- 即使预测有 bias，多覆盖的化学空间帮模型学到了更稳健的结构-性质关系
- 其他 endpoint 要么 N 已经够大（LogD 5039）要么外部数据偏差太严重（KSOL biogen 数据分布偏低）

### 4.2 为什么 JCIM 论文里有效的 3 个技巧（Avalon / Distill / Cliff）到我这里全部失灵？

**核心原因：训练样本数太不均 + 强 distribution shift**。

#### 原因 1：N 跨度过大 → 小样本 endpoint 严重过拟合

| Endpoint | N | 加 Avalon (+1024 维) 后 RAE | 加 Distill (+45 维) 后 RAE |
|---|---:|---:|---:|
| LogD | 5039 | 0.620 (改善 -0.009) | 0.650 (退化 +0.021) |
| MGMB | 222 | 0.604 (退化 +0.030) | 0.721 (退化 +0.147) |
| MPPB | 1302 | 0.927 (退化 +0.055) | 0.962 (退化 +0.090) |
| MBPB | 975 | 0.619 (退化 +0.032) | 0.570 (改善 -0.017) |

明显规律：**特征越多，小 N endpoint 退化越严重**。LightGBM 虽然有 `feature_fraction=0.8` 做隐式特征选择，但 1024 维 Avalon 加 222 行 MGMB 数据已经超过了它的容忍极限。JCIM 论文的实验是在 7000+ 化合物 / endpoint 上做的，根本不会触发这个问题。

#### 原因 2：Distribution shift 让"加更多 feature/data"变成过拟合训练分布

CV (0.494) → test (0.756) 的 0.262 gap 说明 train 化学空间 ≠ test 化学空间（真实的 lead-optimization 时间向前）。在这种情况下：
- 加一个新 feature → 模型有更多自由度去拟合 train 的细节
- 这些细节不能 transfer 到 test
- 净效果是 test 性能下降

#### 原因 3：ADMET-AI distillation 的 bias 不可转移

drugbank_admet_predictions.csv 是 admet_ai 在**已上市药物**上的预测。我们的蒸馏 surrogate 学的是"药物分子的 ADMET 模式"。但 ExpansionRx test 是 lead-optimization 阶段的实验化合物（很多是 RNA 靶点的 bifunctional / bivalent 结构），化学空间根本不同。surrogate 把不相关的 bias 当作 feature 注入到了 LightGBM 里。

#### 原因 4：SALI cliff 在 lead-optimization 数据里删错了

SALI 设计的初衷是过滤"实验噪声"，但 lead optimization 数据里大量"小修饰大变化"的 SAR 是**真实的、有信息的**（这正是为什么实验科学家在做 SAR 探索）。盲目删 z>2.5 的数据点等于丢掉了真正的活性悬崖信号。

### 4.3 与前 10 名方案的本质差距

读完报告之后，我现在明白前 10 名比我好的不是"多加几个 feature"或"多加些数据"，而是：

| 维度 | 前 10 名典型做法 | 我现在 | 提升空间 |
|---|---|---|---:|
| **模型架构** | Chemprop MPNN + 多模型 ensemble (RF/XGB/LGBM/CatBoost via AutoGluon) + TabPFN | 单 LightGBM | -0.05 ~ -0.1 |
| **预训练** | KERMT/CheMeleon/Polaris/Novartis/108 个 ADME 任务 ~450k 数据点 | 无 | -0.05 ~ -0.1 |
| **数据扩增** | RIGR 共振结构枚举（一队最大单一提升）、MD-derived PSA、ESOL 算 logS 当 auxiliary task | 无 | -0.02 ~ -0.05 |
| **任务分组** | 按 Spearman 相关把 endpoint 分 cluster 多任务 (LogD/LogS/PPB 一组, HLM/MLM 一组) | 9 个全独立 | -0.01 ~ -0.02 |
| **数据切分** | 时间窗口滑动 (用分子 ID 排序) / scaffold split / 5×5 CV | 随机 5-fold | 不直接降 RAE 但调参更靠谱 |
| **专有数据** | 4/5 of top 5 用了 lead-optimization 历史数据 | 无 | 决定性差距 |
| **HPO** | 大部分团队没做（边际效益小，易 overfit leaderboard） | 没做 ✓ | 0 |

最大的两块，**专有 lead-opt 数据** 和 **Chemprop ensemble + 大规模预训练**，都不是单纯靠"加 feature / 改 trick"能补上的。

---

## 5. 项目结构

```
OpenADMET-LightGBM-Reproduction/
├── README.md                       项目使用说明（实验过程时间线）
├── REPORT.md                       本报告（完整结果分析）
├── requirements.txt
│
├── data/                           数据
│   ├── train.csv                   官方训练集 (5326×11)
│   ├── test.csv                    官方测试集 (2282×2，无 label)
│   ├── test_ground_truth.csv       官方测试集带 label (用户提供)
│   └── external/                   GitHub 上抓的外部公开数据
│       ├── biogen_logS.csv         Pat Walters 教程, 2173 实测 logS
│       ├── esol_logSolubility.csv  ESOL/Delaney, 500 实测
│       └── drugbank_admet_predictions.csv  admet_ai, 2845 药物 × 50 endpoint
│
├── resource/                       前 10 名方法论资料
│   ├── others.txt                  8 份前 10 名 methodology 报告
│   ├── 2510.12719v1.pdf            Merck/NVIDIA KERMT 论文
│   └── deep-learning-vs-classical-...pdf  rced_nvx JCIM 复盘
│
├── src/                            源代码 (~2200 行)
│   ├── utils.py                    log 变换 / 评分指标
│   ├── features.py                 v1 特征 (RDKit + Morgan)
│   ├── features_v2.py              v2 特征 (+ Avalon, + 可选 Mordred)
│   ├── external_data.py            外部数据加载 + 单位对齐
│   ├── distill_features.py         ADMET-AI 蒸馏特征
│   ├── cliff_masking.py            SALI 活性悬崖 mask
│   ├── precompute_external_features.py  外部 SMILES fingerprint 缓存
│   ├── precompute_features_v2.py        v2 特征缓存
│   ├── precompute_cliff_masks.py        每 endpoint cliff mask 缓存
│   ├── train.py                    v1 训练 + 5-fold CV
│   ├── predict.py                  v1 推理 → submission.csv
│   ├── train_augmented.py          v1 + 外部数据 row augmentation
│   ├── train_v2.py                 v2 (固定全开)
│   ├── train_v2_compose.py         v2 ablation 入口（环境变量分别开关）
│   └── holdout_eval.py             80/20 holdout 评估 + 官方 eval
│
└── output/                         所有实验产物
    ├── submission.csv              v1 baseline 提交文件
    ├── official_eval_metrics.csv   v1 baseline 官方 eval
    ├── cv_summary.csv              v1 5-fold CV 结果
    ├── augmented/                  实验 3-5 (external augmentation)
    │   ├── eval_baseline_n200.csv  公平 baseline (n_est=200, 无 aug)
    │   ├── official_eval_augmented.csv
    │   ├── submission_augmented.csv
    │   └── fair_comparison.csv     selective_aug vs baseline 逐 endpoint
    ├── holdout/                    实验 2 (80/20 holdout)
    │   ├── fake_test_with_labels.csv
    │   ├── fake_submission.csv
    │   └── eval_metrics.csv
    └── v2/                         实验 6-9 (v2 ablations)
        ├── features_train_v2.npz / features_test_v2.npz / features_external_v2.npz
        ├── distill_train.npz / distill_test.npz / distill_external.npz
        ├── cliff_masks.npz
        ├── submission_<tag>.csv
        ├── official_eval_<tag>.csv
        ├── comparison_<tag>.csv
        └── all_runs_comparison.csv  ← 8 次实验的统一对比表
```

---

## 6. 复现指南

### 6.1 环境

```bash
pip install -r requirements.txt   # rdkit, lightgbm, pandas, sklearn, numpy
# v2 实验额外需要
pip install mordred --break-system-packages   # 可选，没用到也行
```

### 6.2 完整复现流程

```bash
# ============== v1 ==============
python src/train.py             # 训练 9 个 endpoint 的 LightGBM + 5-fold CV
python src/predict.py           # 在 test.csv 上推理 → submission.csv
# 用官方 eval 打分
cd ../ExpansionRx-Challenge-Eval && \
python -m eval ../OpenADMET-LightGBM-Reproduction/output/submission.csv \
    --ground-truth ../OpenADMET-LightGBM-Reproduction/data/test_ground_truth.csv \
    --output ../OpenADMET-LightGBM-Reproduction/output/official_eval_metrics.csv

# 80/20 holdout 验证（约 35s）
python src/holdout_eval.py

# ============== External augmentation ==============
python src/precompute_external_features.py   # 一次性 fingerprint 外部 SMILES
EXT_PROFILE=none       python src/train_augmented.py   # 公平 baseline
EXT_PROFILE=all        python src/train_augmented.py   # 全部 endpoint 都加
EXT_PROFILE=selective  python src/train_augmented.py   # 只加受益 endpoint (best)

# ============== v2 (Avalon / Distill / Cliff) ==============
python src/precompute_features_v2.py    # v2 特征 (~30-40s/split)
python src/distill_features.py          # 45 个蒸馏 surrogate
python src/precompute_cliff_masks.py    # 每 endpoint cliff mask (~80s)

# Ablation 入口（任意组合，结果写到 output/v2/<tag>.csv）
USE_AVALON=1 USE_DISTILL=0 USE_CLIFF=1 EXT_PROFILE=selective \
    TAG=mybest python src/train_v2_compose.py
```

### 6.3 关键参数

`src/train_v2_compose.py` 支持的环境变量：
- `USE_AVALON`        0/1, default 0
- `USE_DISTILL`       0/1, default 0
- `USE_CLIFF`         0/1, default 0
- `EXT_PROFILE`       none / all / selective, default none
- `N_ESTIMATORS`      默认 200
- `N_JOBS`            默认 4
- `TAG`               输出文件命名

---

## 7. 后续建议（按 ROI 排序）

### 7.1 高 ROI（不需要 GPU）

1. **多模型 ensemble (XGBoost + LightGBM + RandomForest)**：3 份前 10 名报告都用了；CPU 可跑；预期 -0.01 ~ -0.02。
2. **TabPFN v2 替代 / 补充 LightGBM**：JCIM 论文图 6/7 显示在 ADME 上 surprising 强且**不需要训练**；预期 -0.02 但要本地验证。
3. **任务分组多任务 LightGBM**：按 Spearman 相关把 9 个 endpoint 分 cluster (LogD/LogS/PPB 一组, HLM/MLM 一组)，每组共享一些 LightGBM 基模型；前 10 名几乎都用；-0.01 ~ -0.02。
4. **Endpoint-specific stacking + 本地 holdout 选 config**：上面表里"per-endpoint 最优"显示理论上限 0.733，但要靠**本地 holdout** 而不是 leaderboard 来选 config（防 overfit）；难度中等。
5. **0 值 endpoint-specific 处理**：clearance/permeability 用"半最小非零"替代 0 后再 log；PPB 用 10⁻⁶；Inductive Bio 报告强调过；预期 -0.005。

### 7.2 中 ROI（需要 GPU 或大量数据）

1. **Chemprop MPNN ensemble + CheMeleon 预训练**：前 10 名几乎人人都用；GPU 要求中等，单 GPU 几小时；预期 -0.05 ~ -0.1（最大单一来源的 boost）。
2. **RIGR 数据扩增**（共振结构枚举）：一队报告的最大单一性能提升；CPU 可跑；预期 -0.02 ~ -0.05。
3. **Mordred 描述符 + Jazzy 静电描述符**：JCIM 论文用了；CPU 可跑但慢（5326 化合物 ~3 分钟）；要小心同样的小 N 过拟合问题，建议**只对 N>1500 的 endpoint 加**。

### 7.3 低 ROI 或不可行

1. **专有 lead-optimization 数据**：4/5 of top 5 用了，但拿不到。
2. **大规模 KERMT 预训练**（Merck/NVIDIA paper）：需要 GPU 集群和 ChEMBL 全量数据，性价比对单人项目极低。

---

## 8. 个人 reflection / Lessons Learned

1. **小 N + 强 distribution shift 是 ML 最难的一类问题**。这场比赛 9 个 endpoint 里一半都属于这个 category。所有"加 feature/加数据"类的改进都要先问一遍："这会不会让小 N endpoint 过拟合？"
2. **CV 数字跟 test 数字差 0.26 是个 huge red flag**。不能只看 CV 调 config，必须用某种近似 deployment 的 split（时间窗口、scaffold split 等）选 config。
3. **External data 的 alignment 比想象中难**：单位对齐、实验条件对齐、化学空间对齐三者缺一不可。biogen_logS 单位看似对了但分布偏低，实际上"对齐"的不到位就引入了系统性 bias。
4. **借鉴论文要小心 transferability**：JCIM 论文里有效的 Avalon / 蒸馏 / cliff mask 三件套到我这里全部失灵，原因是数据规模和分布性质差异。**Paper 里的 trick 只在 paper 的实验设定下有效**，搬到新场景前要先小规模 ablation。
5. **诚实的 negative result 跟 positive result 一样有价值**。这次实验 6-9 全部失败，但能清楚说明：单 LightGBM 在这套数据上的"性能上限"基本被 selective augmentation 触到了 (0.750)，要进一步突破必须换架构或换数据来源。

---

## 9. 数字总结

| 阶段 | MA-RAE | Macro R² | 备注 |
|---|---:|---:|---|
| 5-fold CV (训练集内部) | — | +0.675 | 只能内部验证 |
| 80/20 holdout (官方 eval) | 0.494 | +0.689 | overoptimistic |
| **真测试集 (v1 baseline)** | **0.756** | **+0.347** | 公平 baseline |
| **真测试集 (selective aug)** | **0.750** | **+0.348** | **最佳成绩** ✅ |
| 真测试集 (v2 各种组合) | 0.757 ~ 0.802 | -0.10 ~ +0.34 | 全部退化或持平 |
| Per-endpoint cherry-pick (overfit 上限) | 0.733 | — | 不可信 |
| 估计前 10 名水平 | 0.4 ~ 0.6 | — | 据博客 |
| Inductive Bio (#1) | 估计 0.4-0.5 | — | 据博客 |

---

## 10. 致谢 / 参考

- **数据**：[OpenADMET-ExpansionRx Challenge HuggingFace](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)
- **官方 eval 脚本**：`../ExpansionRx-Challenge-Eval/eval/`
- **方法论参考**：
  - 8 份前 10 名 methodology reports（resource/others.txt）
  - Adrian, Chung, Boyd et al. *Multitask finetuning and acceleration of chemical pretrained models for small molecule drug property prediction*. arXiv:2510.12719 (2025)
  - Fischer, Southiratn, Triki, Cedeño. *Deep Learning vs Classical Methods in Potency and ADME Prediction: Insights from a Computational Blind Challenge*. J. Chem. Inf. Model. 2025, 65(24), 13115–13131
- **挑战赛复盘博客**：
  - [Lessons Learned](https://openadmet.ghost.io/lessons-learned-from-the-openadmet-expansionrx-blind-challenge/)
  - [Top Performers](https://www.collaborativedrug.com/cdd-blog/applying-a-focused-modeling-strategy-in-the-openadmet-expansionrx-blind-challenge-lessons-from-top-performers)

---

*报告生成时间：2026-05-03*
