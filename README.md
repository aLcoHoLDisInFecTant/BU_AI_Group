# Sentiment Analysis Project (IMDb)

本项目实现一个可复现的情感分析实验框架，支持 RNN（Bi-LSTM/GRU）与 BERT 两种模型，并通过配置文件驱动实验，满足“配置驱动”“路径可配置”“可复现性”的要求。

## 环境与依赖

- Python >= 3.8（已在本项目中使用 `3.8.x`）
- 建议使用 GPU（CUDA 11.8 + cuDNN），示例环境：`conda env pytorch-gpu-11.8`

安装依赖：

```
pip install -r requirements.txt
```

激活 GPU 环境（Windows PowerShell）：

```
conda activate pytorch-gpu-11.8
```

如遇 SSL 证书问题（下载 Hugging Face 资源失败），可启用离线模式使用本地缓存：

```
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
```

## 数据

- 请确保已有 IMDb 数据集 CSV 文件（如 Kaggle 的 `IMDB Dataset.csv`）。
- 本仓库默认从上级目录 `../data/IMDB Dataset.csv` 读取数据（与你当前工作区的 `d:\pythonProject\015assignment\data\IMDB Dataset.csv` 对齐）。
- 如路径不同，请在配置文件中修改 `csv_path`。

可选：如需使用 GloVe 预训练词向量，请下载 `glove.6B.100d.txt` 到 `../data/glove/`，并在对应 RNN 配置文件中设置 `glove_path`。

## 项目结构

```
sentiment-analysis-project/
├── README.md
├── requirements.txt
├── configs/
│   ├── rnn_random_embed.json
│   ├── rnn_glove_frozen.json
│   ├── rnn_glove_finetune.json
│   ├── bert_freeze.json
│   └── bert_finetune.json
├── src/
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── main.py
│   └── utils.py
├── analysis/
│   ├── 1_plot_learning_curves.ipynb
│   ├── 2_compare_models.ipynb
│   └── 3_error_analysis.ipynb
├── data/                # (gitignored) - 可选本地缓存目录
├── models_checkpoints/  # (gitignored)
└── results/             # (gitignored)
```

## 运行示例

在项目根目录下运行（确保使用目标 Python/Conda 环境）：

```
# RNN系列
python src/main.py --config configs/rnn_random_embed.json
python src/main.py --config configs/rnn_glove_frozen.json
python src/main.py --config configs/rnn_glove_finetune.json

# BERT系列
python src/main.py --config configs/bert_freeze.json   # 冻结全部或多数层
python src/main.py --config configs/bert_finetune.json # 微调更多层
python src/main.py --config configs/bert_fast.json     # 加速版配置（低max_length/少解冻层）

# 正式配置示例（较长训练）
python src/main.py --config configs/bert_formal.json
```

说明：
- 所有实验均通过配置文件驱动，包括模型类型、超参数、数据路径、随机种子等。
- 日志与指标会保存到 `results/` 文件夹（如 `results_seed_42_rnn.json`）。
- 模型权重会保存到 `models_checkpoints/`。

## 配置文件关键项

- `model_type`: `"rnn"` 或 `"bert"`
- `embedding_mode`: `"random"` / `"glove_frozen"` / `"glove_finetune"`
- `seed`: 随机种子（建议使用 42、60、70 进行三次独立实验）
- `csv_path`: IMDb 数据集 CSV 的路径（默认 `../data/IMDB Dataset.csv`）
- `sample_size`: 采样大小（读取部分数据以加速开发）
- `max_length`: 序列最大长度（RNN 与 BERT 可分别配置）
- 其他超参数：学习率、批大小、dropout、隐藏层维度、双向与否、BERT 解冻层数等。

BERT 冻结/微调相关：
- `freeze_all`: `true/false`，是否冻结全部 BERT 参数。
- `unfreeze_layers`: 解冻的后向层数，如 `2` 表示解冻最后两层。

## 结果分析（analysis/）

- `1_plot_learning_curves.ipynb`: 加载 `results/` 中的训练日志，绘制训练/验证 Loss 与 Accuracy 曲线。
- `2_compare_models.ipynb`: 聚合不同模型、不同种子的 P/R/F1 进行对比，输出表格与可视化。
- `3_error_analysis.ipynb`: 对最佳模型的测试集预测进行误差桶分析（否定词、讽刺、长度分组等）。

## 复现实验建议

- 固定随机种子（`utils.set_seed`）。
- 仅用训练集构建词表，处理 OOV。
- RNN：根据配置选择随机嵌入或加载 GloVe，并选择冻结/微调策略。
- BERT：根据配置选择冻结全部或仅解冻最后若干层。

## 日志与输出（可读性增强）

- 训练日志现在包含清晰的轮次边界：`===== Epoch X/Y =====`。
- 每轮结束打印训练/验证汇总：`loss` 与 `acc`（准确率）。
- 验证与测试阶段会额外打印混淆矩阵，以及按文本长度分桶的准确率/错误率，方便快速定位问题场景。
- 完成全部轮次后打印测试集汇总并保存产物：
  - 结果：`results/results_seed_<seed>_<model>.json`
  - 权重：`models_checkpoints/model_seed_<seed>_<model>.pt`
- tqdm 进度条在每轮结束后保留，便于回看每轮耗时与速度。

## 常见问题

- SSL 错误：启用离线模式（见上文），或检查本地证书与网络代理。
- 速度慢：使用 `configs/bert_fast.json`，或降低 `max_length`、`sample_size`。
- GPU 未启用：确认 `conda activate pytorch-gpu-11.8`，以及 `torch.cuda.is_available() == True`。

## 注意事项

- 初次运行建议使用较小的 `sample_size` 与较短的 `max_length` 以验证流程。
- 如需完整训练，请取消采样并增大训练轮数与长度。