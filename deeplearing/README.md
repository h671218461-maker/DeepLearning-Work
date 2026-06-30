# 深度学习课程设计：scRNA-seq 细胞类型注释大模型

本项目实现一个面向单细胞 RNA 测序数据的细胞类型注释模型 `scCAT`
（single-cell Cell Annotation Transformer）。项目结构参考课程设计模板：

```text
课程设计/
├── data/scrna/              # 数据说明与真实数据放置位置
├── model/                   # Baseline 与 Transformer 大模型
├── utils/                   # 数据预处理、指标、可视化
├── .gitignore
├── DL课程设计.ipynb          # 完整课程设计报告与实验代码
├── README.md
├── requirements.txt
└── train.py
```

## 研究任务

输入为单细胞基因表达矩阵，输出为每个细胞的类型标签，例如 T cell、B cell、
monocyte、NK cell 等。本项目把每个细胞表示为“高表达基因 token 序列”，并将
基因 ID embedding、表达量 embedding 和位置 embedding 相加后送入 Transformer
Encoder，最后使用 `[CLS]` token 做细胞级分类。

## 模型设计

- Baseline：小型 MLP，输入完整表达向量，用作传统深度学习对照。
- scCAT：基因 token + 表达量数值编码 + 多层 Transformer Encoder。
- 训练目标：监督交叉熵分类；代码中保留表达量重建头，可扩展为 masked
  expression modeling 预训练。
- 评价指标：accuracy、macro precision、macro recall、macro-F1、混淆矩阵。

## 快速运行

```bash
pip install -r requirements.txt
python train.py --synthetic-smoke-test --epochs 3 --batch-size 32
```

使用真实 CSV 数据：

```bash
python train.py --data-csv data/scrna/pbmc.csv --label-col cell_type --epochs 20
```

训练结果会保存到 `outputs/`，包括模型权重、训练历史和测试集指标。

## 说明

`DL课程设计.ipynb` 已按课程报告模板补全，包含摘要、问题定义、数据集说明、
模型结构、实验设计、可视化分析、总结与展望。若需要提交课程报告，建议在
notebook 中填写个人姓名、学号、班级、指导教师和提交日期，并根据实际运行
结果更新实验表格。
