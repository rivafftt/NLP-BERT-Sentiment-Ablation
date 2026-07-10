# NLP-BERT-Sentiment-Ablation
基于 BERT 与传统机器学习的中文情感分类对比实验与消融分析 (ChnSentiCorp 数据集)
# NLP-BERT-Sentiment-Ablation

> **课程大作业**：基于 BERT 与传统机器学习的中文情感分类对比实验与消融分析

本项目聚焦于中文文本情感二分类任务，基于公开的 **ChnSentiCorp** 数据集展开全流程的系统化建模与深层对比。通过结合传统特征工程基线与深度学习端到端微调方案，探索预训练权重与微调策略对大语言模型泛化性能的深层贡献。

---

## 实验轨道与消融结果对照

| 实验组别 / 方案名称 | 模型底层配置 | 拆解组件公式 | 测试集准确率 (Acc) | 相对基线变化 |
| :--- | :--- | :---: | :---: | :---: |
| 传统机器学习基线 | TF-IDF + 逻辑回归 | 统计学词袋 | 86.58% | 对照基准 |
| 消融对照组 1 | BERT 纯随机初始化训练 | $A + C$ | 86.17% | -0.41% |
| 消融对照组 2 | BERT 骨干网络参数冻结 | $A + B$ | 76.08% | -10.50% |
| **本文最优解 (完全体)** | **BERT 全参数微调方案** | $\mathbf{A + B + C}$ | **93.00%** | **+6.42%** |

> *注：组件 A 表示 Transformer 网络架构，B 表示预训练权重，C 表示本地全参数微调。*

---

## 仓库目录结构说明

```text
├── train.tsv                   # 训练集 (9,146 条)
├── dev.tsv                     # 验证集
├── test.tsv                    # 测试集 (1,200 条)
├── baselines/                  # 传统机器学习基线轨道
│   └── baseline_ml.py          # TF-IDF + LR 核心流水线
├── proposed_bert/              # 主力深度学习方案
│   └── train_bert.py           # BERT 全参数端到端微调脚本
├── ablation_studies/           # 严谨学术消融实验
│   ├── train_bert_scratch.py   # 消融 1：随机初始化训练
│   └── train_bert_frozen.py    # 消融 2：主干网络参数冻结
├── outputs_and_analysis/       # 实验输出与深度定性分析 (加分项)
│   ├── ml_error_cases.csv      # 传统机器学习基线预测错误的 161 条“错题本”
│   ├── bert_loss_curve.png     # BERT 微调过程中的损失函数收敛趋势图
│   └── training_metrics.log    # 各轨道模型训练全量日志与吞吐量记录
└── requirements.txt            # 项目最小核心环境依赖 （本次实验我使用的是Python 3.11.12）

