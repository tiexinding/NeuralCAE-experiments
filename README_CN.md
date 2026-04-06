# NeuralCAE 实验仓库

一个CAE工程师的物理AI探索：MeshGraphNet、FNO、PyG实验，以及NPM（神经渗流模型）验证。

[English README](README.md)

---

## NPM框架研究系列

| # | 主题 | 状态 | 链接 |
|---|------|------|------|
| #1 | 涌现排序预测 | 已发布 | [Zenodo (DOI: 10.5281/zenodo.19440042)](https://doi.org/10.5281/zenodo.19440042) |
| #2 | 临界指数的物理解释 | 计划中 | — |
| #3 | 跨临界点涌现曲线 | 计划中 | — |
| #4 | 数据质量参数的操作化定义 | 计划中 | — |

**NPM主框架：** [Zenodo (DOI: 10.5281/zenodo.19209722)](https://doi.org/10.5281/zenodo.19209722)

### 验证数据与代码（`validation/`）

- `scripts/` — NPM拟合、Brier Score计算脚本
- `results/` — 9个模型的MMLU评测结果（Pythia 70M–12B + Phi-2）
- `figures/` — Accuracy vs Brier Score散点图（中英文版）

---

## MeshGraphNets实验（基于PhysicsNeMo）

NVIDIA PhysicsNeMo框架下MeshGraphNet模型的API验证、训练流程复现与文档问题记录。

> 基于DeepMind 2021年发表的 [MeshGraphNets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) 论文，使用NVIDIA PhysicsNeMo框架提供的模型实现。

### 目录结构

```
NeuralCAE-experiments/
├── 01-meshgraphnets.py          # PhysicsNeMo API测试（DGL后端）
├── 01-meshgraphnets-3.py        # PhysicsNeMo API测试（PyG后端）+ Bug报告
├── 01-MeshGraphNets_PyG.py      # 完整训练流程（PyG，基于DeepMind论文）
├── 02-pyg.py                    # DGL vs PyG图创建方式对比
├── validation/                  # NPM验证实验
│   ├── scripts/                 # NPM拟合与Brier Score脚本
│   ├── results/                 # MMLU评测结果（9个模型）
│   └── figures/                 # 散点图（中英文版）
├── datasets/                    # 数据集（.gitignore排除）
├── best_models/                 # 训练产出
├── animations/                  # 速度场预测动画（GIF）
└── 2d_loss_plots/               # Loss曲线图（PDF）
```

### 发现的问题

测试PhysicsNeMo MeshGraphNet API过程中，发现NVIDIA官方文档存在3个问题：

1. **输入类型错误** — 文档示例传递Tensor，实际需要PyG Data对象
2. **HybridMeshGraphNet边结构** — 文档未说明需要合并mesh edges和world edges
3. **BiStrideMeshGraphNet示例不可用** — 需要专用数据集类预处理

### 使用方法

```bash
# API测试（推荐PyG后端）
python 01-meshgraphnets-3.py

# 完整训练流程（需要datasets/valid.h5）
python 01-MeshGraphNets_PyG.py
```

---

**许可证：** MIT | **作者：** 丁铁新 | **NeuralCAE** | 所有工作均利用业余时间完成。
