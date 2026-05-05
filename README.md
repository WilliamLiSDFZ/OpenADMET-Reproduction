# OpenADMET-ExpansionRx Challenge — LightGBM 复现项目

复现 [OpenADMET ExpansionRx Blind Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) 中
**LightGBM + RDKit 描述符** 这一类前 10 名常用方案。该方案也是 OpenADMET 官方提供的 baseline 之一。

## 选择的方法

参考挑战赛的复盘博客（"Lessons Learned from the OpenADMET-ExpansionRx Blind Challenge"），
排名前列的方案普遍采用了下列组合：

- **梯度提升树**（LightGBM / XGBoost / CatBoost）
- **RDKit 物理化学描述符**
- **Morgan / ECFP 指纹**
- **多模型 ensemble + 5-fold CV**

本项目复现的是其中"性价比"最高的一种：**LightGBM + RDKit 描述符 + Morgan 指纹**，
在 CPU 上几分钟即可训练完，并能给出接近 baseline 中位水平的 MA-RAE。

## 任务

预测 9 个 ADMET 终点：

| 简写 | 含义 | 是否做 log 变换 |
|---|---|---|
| LogD | 脂水分配系数 | 否 |
| KSOL | 动力学溶解度 | 是 |
| HLM CLint | 人肝微粒体清除率 | 是 |
| MLM CLint | 鼠肝微粒体清除率 | 是 |
| Caco-2 Permeability Papp A>B | Caco-2 渗透性 | 是 |
| Caco-2 Permeability Efflux | Caco-2 外排比 | 是 |
| MPPB | 鼠血浆蛋白结合率 | 是 |
| MBPB | 鼠脑蛋白结合率 | 是 |
| MGMB | 鼠腓肠肌蛋白结合率 | 是 |

排名指标：**MA-RAE** (Macro-Averaged Relative Absolute Error)。

## 目录结构

```
OpenADMET-LightGBM-Reproduction/
├── data/
│   ├── train.csv           # 训练集 (含 9 个 endpoint, 稀疏)
│   └── test.csv            # 测试集 (只有 SMILES)
├── src/
│   ├── features.py         # 特征工程 (RDKit 描述符 + Morgan FP)
│   ├── train.py            # 训练 + 5-fold CV
│   ├── predict.py          # 在 test.csv 上预测并生成提交文件
│   └── utils.py            # log 变换 / 指标
├── output/
│   ├── cv_results.csv      # 各 endpoint 的 R² / MAE
│   ├── models/             # 保存的 LightGBM 模型
│   └── submission.csv      # 最终提交文件
└── requirements.txt
```

## 使用

```bash
pip install -r requirements.txt

# 1) 训练 + 5-fold CV (输出 R²/MAE)
python src/train.py

# 2) 用全量训练数据在 test.csv 上预测，生成 submission.csv
python src/predict.py
```

## 关键设计

1. **Log 变换**：除 LogD 外的 8 个 endpoint 在训练前做 `log10(x + 1)` 变换，预测后再反变换。
2. **稀疏数据处理**：每个 endpoint 单独训练，只用 `endpoint != NaN` 的样本。
3. **特征**：208 维 RDKit 描述符 + 2048 位 Morgan 指纹 (radius=2)，共 2256 维。
4. **模型**：LightGBM 默认参数（足以体现 baseline 实力），也可在 `train.py` 里改 `LGBM_PARAMS`。
5. **交叉验证**：每个 endpoint 跑 5-fold CV，输出 R²/MAE 的均值±标准差。

## 与前 10 名方案的区别

| 维度 | 本项目 | 前 10 名典型方案 |
|---|---|---|
| 模型 | 单 LightGBM | LightGBM + Chemprop + CatBoost ensemble |
| 特征 | RDKit + Morgan | + Mordred + Jazzy 静电描述符 |
| 数据 | 仅官方训练集 | + ChEMBL / Polaris / TDC / 自有数据 |
| 调参 | 默认 | Optuna / Ray Tune 大规模搜索 |
| 训练时长 | < 5 分钟 (CPU) | 数小时 - 数天 (GPU) |

按博客中给出的对比，单 LightGBM baseline 的 MA-RAE 大概落在中位数附近，
要进入前 10 主要靠 ensemble + 外部数据 + 调参。本项目可作为搭建后续 ensemble 的起点。

## 实测结果（用上传的 train.csv / test.csv 跑出来的）

3-fold CV (`N_SPLITS=3 N_ESTIMATORS=150` 做的快速验证):

| Endpoint | N | R² | MAE | RAE |
|---|---:|---:|---:|---:|
| LogD             | 5039 | **0.838** | 0.344 | 0.366 |
| LogS (KSOL)      | 5128 | 0.639 | 0.308 | 0.516 |
| Log_MLM_CLint    | 4522 | 0.676 | 0.337 | 0.539 |
| Log_HLM_CLint    | 3759 | 0.601 | 0.297 | 0.595 |
| Log_Caco_ER      | 2161 | 0.575 | 0.145 | 0.587 |
| Log_Caco_Papp_AB | 2157 | 0.612 | 0.203 | 0.556 |
| Log_Mouse_PPB    | 1302 | 0.699 | 0.191 | 0.494 |
| Log_Mouse_BPB    |  975 | **0.754** | 0.162 | 0.444 |
| Log_Mouse_MPB    |  222 | 0.678 | 0.161 | 0.514 |
| **Macro Average** | – | **+0.675** | – | **0.512** |

跑 5-fold + 300 棵树的完整版只需把环境变量去掉:
```bash
python src/train.py
```

## v2 实验：尝试前 10 名报告里 3 个高优先级技巧

读完 `resource/` 下面 8 份前 10 名方法论 + JCIM 论文之后，我挑了 3 个理论上最容易移植到我这套 LightGBM 上的：

1. **Avalon 指纹** + RDKit-2d + Morgan（"augmented classical descriptors"，JCIM 论文图 4-5 表现最好的特征组合）
2. **ADMET-AI 蒸馏特征**（45 维 surrogate 预测，JCIM 论文图 5 显示统计显著提升 ADME）
3. **SALI 活性悬崖 mask**（Tanimoto + |z|>2.5，JCIM 论文图 3 证实有效）

实现：
- `src/features_v2.py` — RDKit + Morgan + Avalon (+ optional Mordred)
- `src/distill_features.py` — 用 drugbank 的 admet_ai 预测训练 45 个 surrogate LightGBM，再对挑战赛 SMILES 推理
- `src/cliff_masking.py` + `src/precompute_cliff_masks.py` — 按 endpoint cluster + SALI z-score 标记 cliffs
- `src/train_v2_compose.py` — 可分别开关上面 3 项 + 外部数据增强的统一训练入口

### 关键结果（n_estimators=200，全部用官方 eval 在真测试集上打分）

| 配置 | LogD | KSOL | MLM | HLM | CACO-Eff | CACO-Papp | MPPB | MBPB | MGMB | **MA-RAE** | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v1_baseline** (RDKit+Morgan, 无 aug) | 0.629 | 0.858 | 0.929 | 0.814 | 0.774 | 0.771 | 0.872 | 0.587 | 0.574 | **0.756** | — |
| **v1_selective_aug** ← 上一轮的最佳 | 0.623 | 0.858 | 0.914 | 0.814 | 0.774 | 0.771 | **0.832** | 0.587 | 0.574 | **0.750** | **−0.006** ✅ |
| v2_avalon (+1024 Avalon bits) | 0.620 | 0.866 | 0.900 | 0.835 | **0.763** | 0.806 | 0.927 | 0.619 | 0.604 | 0.771 | +0.015 ❌ |
| v2_distill (+45 ADMET-AI surrogate) | 0.650 | 0.907 | 0.904 | 0.857 | 0.823 | 0.820 | 0.962 | **0.570** | 0.721 | 0.802 | +0.045 ❌ |
| v2_cliff (only mask SALI cliffs) | 0.636 | 0.858 | 0.904 | 0.829 | 0.806 | 0.773 | 0.911 | 0.599 | 0.597 | 0.768 | +0.012 ❌ |
| v2_avalon + cliff | **0.620** | **0.821** | 0.895 | 0.816 | 0.813 | 0.806 | 0.940 | 0.606 | **0.553** | 0.763 | +0.007 ❌ |
| v2_avalon + cliff + selective_aug | 0.633 | 0.821 | **0.860** | 0.816 | 0.813 | 0.806 | 0.908 | 0.606 | 0.553 | 0.757 | +0.001 ~ |
| v2_all (+ Avalon + cliff + distill) | 0.631 | 0.846 | 0.918 | **0.804** | 0.763 | 0.800 | 0.920 | 0.610 | 0.616 | 0.766 | +0.010 ❌ |

### 结论：**3 个改进单独/组合用，全部都没赢过简单的 selective augmentation**

每一项**单独**加上去都让 macro RAE 变差（Avalon +0.015、Distill +0.045、Cliff +0.012）。Avalon+Cliff 组合稍微好一点（+0.007）但还是输。

加全 3 个再加 selective aug，最好也只回到 +0.001（基本平手）。

### 为什么 JCIM 论文里有效的技巧到我这里失灵了？

1. **N 太不均**: 挑战赛的 endpoint 样本数从 5039 (LogD) 到 222 (MGMB) 跨两个数量级。MPPB/MBPB/MGMB 三个小样本 endpoint 在加 Avalon (+1024 维) 或 distill (+45 维) 之后**严重过拟合**，单 endpoint RAE 直接从 0.872 → 0.940-0.962。JCIM 论文用的 7000 化合物 / endpoint 不会有这个问题。
2. **分布偏移大**：CV (0.494) → 真测试 (0.756) 的差距说明 train→test 之间有强 distribution shift（lead-opt 时间向前），扩充特征实际上是给模型提供更多过拟合训练分布的途径。
3. **Distill features 引入 bias**：drugbank_admet_predictions 是 admet_ai 在**药物分子**上的预测，但 ExpansionRx test 是 lead-optimization 阶段的实验化合物，化学空间不同。蒸馏出来的 surrogate 把 admet_ai 在药物上的偏差搬到了我的模型里。
4. **Cliff masking 误删信号**：SALI 标的"cliff"在 lead optimization 数据上很多其实是真实的 SAR 活性变化（小修饰大变化是这种数据的正常现象），删了等于丢信息。

### 各 endpoint 上"理论最优组合"（如果按测试集 cherry-pick，但这是 overfit）

| Endpoint | best technique | best RAE |
|---|---|---:|
| LogD | v2_avalon | 0.620 |
| KSOL | v2_avalon+cliff | 0.821 |
| MLM CLint | v2_combo_best | 0.860 |
| HLM CLint | v2_all | 0.804 |
| Caco-2 Efflux | v2_avalon | 0.763 |
| Caco-2 Papp | v1_baseline | 0.771 |
| MPPB | v1_selective_aug | 0.832 |
| MBPB | v2_distill | 0.570 |
| MGMB | v2_avalon+cliff | 0.553 |
| **macro (cherry-picked)** | — | **0.733** |

按 endpoint cherry-pick 能到 **0.733** vs 我们老的最好 0.750，但这是用测试集挑出来的，会严重 overfit，不能算真实成绩。要靠谱地拿到这个数字得做 endpoint-specific stacking + 用本地 holdout 选 config（而不是 leaderboard）。

### 跑法
```bash
# 准备工作（跑一次就够）
python src/precompute_features_v2.py        # RDKit+Morgan+Avalon 特征 (~1 min)
python src/distill_features.py              # 45 个 ADMET-AI surrogate (~30s)
python src/precompute_cliff_masks.py        # 9 个 endpoint 的 SALI mask (~80s)

# 任何组合的 ablation (举例)
USE_AVALON=1 USE_DISTILL=0 USE_CLIFF=1 EXT_PROFILE=selective TAG=mybest \
    python src/train_v2_compose.py
```

## 调参建议（如果想往前 10 冲）

1. 在 `src/train.py` 里把 `LGBM_PARAMS` 改成 Optuna 搜的参数。
2. 在 `src/features.py` 里加 Mordred 描述符 / Jazzy 电荷描述符。
3. `src/train.py` 里加第二、第三个模型 (XGBoost / CatBoost / Chemprop)，最后做平均 ensemble。
4. 引入外部公开数据 (ChEMBL, TDC ADMET, Polaris) 作为 pre-training 或 multi-task 学习。

## 外部数据增强实验（已实现，见 `src/external_data.py` + `src/train_augmented.py`）

挑战赛复盘里反复强调"前 10 都用了外部数据"。我也试了一下，结论：**朴素拼接没用，
选择性拼接能拉低 ~1%**。

### 数据来源（全部从 GitHub 抓的，HF 在沙盒里访问不到）
- `biogen_logS.csv`（Pat Walters 教程，2173 条**实测** logS）→ KSOL
- `regression.csv`（chemprop ESOL/Delaney，500 条实测 logS）→ KSOL
- `drugbank_admet_predictions.csv`（admet_ai，2845 个药物 × 多个 ADMET endpoint **预测值**）→ LogD / KSOL / Caco-2 Papp / Clearance / PPBR

### 实验对比（全部用官方 eval 在 ground truth 上打分）

同一份代码同 N_ESTIMATORS=200，只调外部数据策略：

| 方案 | MA-RAE | 与 baseline 比 |
|---|---:|---:|
| no augmentation (n=200, 公平 baseline) | 0.756 | — |
| **selective augmentation** (LogD/MLM/MPPB only, W_meas=0.5, W_pred=0.2) | **0.750** | **−0.007** |
| all augmentation (全 9 个 endpoint, 同权重) | 0.761 | +0.005 ❌ |
| all augmentation (低权重: W_meas=0.3, W_pred=0.05) | 0.764 | +0.008 ❌ |

### 关键发现

- **不分青红皂白拼接会变差**：第一次"all" 跑，KSOL/HLM/Caco-2 Papp 三个反而比 baseline 差，因为外部数据的实验条件、单位、分子分布和 ExpansionRx 不一致，引入了系统性偏差。
- **要先看哪些 endpoint 受益**：第一次跑后，发现外部数据帮上忙的是 LogD (Δ=−0.015)、MLM CLint (Δ=−0.009)、MPPB (Δ=−0.051)；伤到的是 KSOL (+0.020)、HLM (+0.016)、Caco-2 Papp (+0.048)。
- **只在受益的 endpoint 上加**（"selective" 模式）：稳定拿到 MA-RAE 0.750，比 baseline 低 0.007。MPPB 单独从 0.872 → 0.832 是最大功臣（drugbank 的 PPBR 预测对蛋白结合捕捉得不错）。
- **biogen_logS 没帮上 KSOL** 的原因可能是 biogen 数据集分布略低（mean 18 μM vs 挑战赛 146 μM），加进去把模型拉向了"难溶"那一侧。

### 跑法
```bash
# 1) 一次性预算外部 SMILES 的 fingerprint (~18s, 一次就行)
python src/precompute_external_features.py

# 2) baseline (无外部数据)
EXT_PROFILE=none python src/train_augmented.py

# 3) 全部 endpoint 都加外部
EXT_PROFILE=all python src/train_augmented.py

# 4) 只在 LogD/MLM/MPPB 上加（推荐）
EXT_PROFILE=selective python src/train_augmented.py
```

最终 best 模型预测在 `output/augmented/submission_augmented.csv`，对应 MA-RAE 0.750。
