# MeshGraphNets Experiments with PhysicsNeMo

MeshGraphNets 模型在 NVIDIA PhysicsNeMo 框架下的 API 验证、训练流程复现与文档问题记录。

> 实验基于 DeepMind 2021 年发表的 [MeshGraphNets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) 论文，使用 NVIDIA PhysicsNeMo 框架提供的模型实现。

---

## Environment / 环境依赖

| Package | 用途 |
|---------|------|
| PyTorch | 深度学习框架 |
| PyTorch Geometric (PyG) | 图神经网络后端（推荐） |
| DGL | 图神经网络后端（旧版 PhysicsNeMo 需要） |
| PhysicsNeMo | NVIDIA 物理仿真 AI 框架，提供 MeshGraphNet 模型 |
| h5py | 读取 HDF5 格式的 CFD 仿真数据 |
| TensorFlow 1.x | 仅用于数据预处理（`triangles_to_edges` 函数） |
| torch_scatter | PyG 消息传递依赖 |
| matplotlib | 可视化与动画生成 |

---

## Directory Structure / 目录结构

```
test-add/
├── 01-meshgraphnets.py          # PhysicsNeMo API 测试 (DGL backend)
├── 01-meshgraphnets-3.py        # PhysicsNeMo API 测试 (PyG backend) + 文档问题记录
├── 01-MeshGraphNets_PyG.py      # 完整训练流程 (PyG, 基于 DeepMind 论文)
├── 02-pyg.py                    # DGL vs PyG 图创建方式对比
├── datasets/                    # 数据集 (excluded by .gitignore)
│   ├── valid.h5                 # 原始 CFD 仿真数据 (HDF5)
│   ├── test_processed_set.pt    # 预处理后的图数据
│   └── meshgraphnets_miniset*.pt # 不同规模的子集
├── best_models/                 # 训练产出
│   ├── *.pt                     # 模型权重 (excluded by .gitignore)
│   └── *.csv                    # 训练日志 (train/test/velo loss)
├── animations/                  # 速度场预测动画 (GIF)
├── 2d_loss_plots/               # Loss 曲线图 (PDF)
└── .gitignore
```

---

## Scripts / 脚本说明

### 1. `01-meshgraphnets.py` — API Test (DGL Backend)

使用 DGL 图格式测试 PhysicsNeMo 提供的 MeshGraphNet 模型：

- **MeshGraphNet**: 基础模型，100 节点 toy graph 前向传播
- **MeshGraphKAN**: KAN (Kolmogorov-Arnold Networks) 变体，`num_harmonics=5`
- **BiStrideMeshGraphNet**: 多尺度变体，需配合 `BistrideMultiLayerGraphDataset` 使用

```python
from physicsnemo.models.meshgraphnet import MeshGraphNet, MeshGraphKAN, BiStrideMeshGraphNet
dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
model = MeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2, processor_size=10)
output = model(node_features, edge_features, dgl_graph)  # [N, 2]
```

### 2. `01-meshgraphnets-3.py` — API Test (PyG Backend) + Bug Report

使用 PyG `Data` 对象测试全部 4 个模型变体，同时记录了 NVIDIA 官方文档中的问题（详见 [Findings](#findings--发现的问题) 部分）：

- **MeshGraphNet** — 正常工作
- **HybridMeshGraphNet** — 修正后正常工作（需合并 mesh + world edges）
- **MeshGraphKAN** — 正常工作
- **BiStrideMeshGraphNet** — 简单示例无法运行，需要专用数据集类

### 3. `02-pyg.py` — DGL vs PyG Graph Creation

对比两种图框架创建同一个 3 节点有向图的写法：

```python
# DGL:
graph_dgl = dgl.graph((src, dst))
graph_dgl.ndata["x"] = node_features

# PyG:
edge_index = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)
graph_pyg = Data(x=node_features, edge_index=edge_index)
```

### 4. `01-MeshGraphNets_PyG.py` — Full Training Pipeline

基于 DeepMind 论文的**完整 MeshGraphNet 训练流程**，使用 PyG 实现。包含以下模块：

| 模块 | 功能 |
|------|------|
| Data Loading | 从 `valid.h5` 读取 CFD 仿真数据，构建 PyG 图（节点特征=速度+节点类型 one-hot，边特征=相对位置+距离） |
| `ProcessorLayer` | 自定义 MessagePassing 层（edge MLP → node MLP，带残差连接） |
| `MeshGraphNet` | Encoder-Processor-Decoder 架构（多层 ProcessorLayer） |
| `train()` / `test()` | 训练循环 + 加速度 loss + 速度 validation RMSE |
| `visualize()` | 基于 matplotlib 生成 Ground Truth / Prediction / Error 三行动画 |

---

## Experiments / 实验配置

已完成的实验（从模型文件名解析）：

| num_layers | batch_size | hidden_dim | epochs | lr | weight_decay | shuffle | train/test | 备注 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | False | 45/10 | 对照组 |
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | True | 45/10 | shuffle 对比 |
| 10 | 16 | 10 | 5000 | 0.001 | 5e-4 | True | 85/15 | 增大数据量 |
| 10 | 16 | **30** | 5000 | 0.001 | 5e-4 | True | **420/80** | 最大规模实验 |

可视化产出：
- `2d_loss_plots/` — shuffle=True vs False 的 loss 对比曲线
- `animations/` — 不同实验配置下的 x-velocity 速度场预测动画

---

## Findings / 发现的问题

在测试 PhysicsNeMo MeshGraphNet API 过程中，发现 NVIDIA 官方文档存在以下 **3 个问题**：

### Bug 1: 输入类型错误

文档示例传递 `edge_index` (Tensor)，但实际模型需要 PyG `Data` 对象：

```python
# 文档示例 (错误):
model(node_features, edge_features, edge_index)

# 正确用法:
graph = Data(edge_index=edge_index)
model(node_features, edge_features, graph)
```

### Bug 2: HybridMeshGraphNet 边结构

文档未说明 HybridMeshGraphNet 需要将 mesh edges 和 world edges **合并**到同一个图中：

```python
# 正确用法: 合并两组边
combined_edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)
hybrid_graph = Data(edge_index=combined_edge_index)
model_hybrid(node_features, mesh_edge_features, world_edge_features, hybrid_graph)
```

### Bug 3: BiStrideMeshGraphNet 示例不可用

文档中的简单示例无法运行。该模型需要使用 `BistrideMultiLayerGraphDataset` 预处理生成正确的多尺度图结构（`ms_edges`, `ms_ids`），不能手动构造。

---

## Usage / 使用说明

### API 测试

```bash
# DGL backend (需要 DGL 环境)
python 01-meshgraphnets.py

# PyG backend (推荐)
python 01-meshgraphnets-3.py
```

### 完整训练流程

```bash
# 需要 datasets/valid.h5 和 datasets/meshgraphnets_miniset5traj_vis.pt
python 01-MeshGraphNets_PyG.py
```

> 注意：训练部分代码默认被注释（`01-MeshGraphNets_PyG.py:629`），取消注释后即可训练。当前脚本直接加载已训练好的模型权重进行可视化。

---

## Data / 数据说明

数据集文件较大（总计 ~4.9GB），**未包含在 Git 仓库中**。

| 文件 | 大小 | 说明 |
|------|------|------|
| `valid.h5` | 1.2 GB | 原始 CFD 仿真数据（HDF5 格式），包含多条 trajectory 的 velocity, pressure, mesh_pos, cells, node_type |
| `meshgraphnets_miniset5traj_vis.pt` | 1.3 GB | 5 条 trajectory 的预处理图数据（完整时间步） |
| `meshgraphnets_miniset100traj25ts_vis.pt` | 1.1 GB | 100 条 trajectory, 25 个时间步 |
| `meshgraphnets_miniset30traj5ts_vis.pt` | 59 MB | 30 条 trajectory, 5 个时间步（轻量级） |
| `test_processed_set.pt` | 1.8 MB | 2 条 trajectory, 2 个时间步（toy example） |

数据来源：DeepMind MeshGraphNets 论文配套的 cylinder flow 数据集。
