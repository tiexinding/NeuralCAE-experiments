import torch
from torch_geometric.data import Data
from physicsnemo.models.meshgraphnet import MeshGraphNet, HybridMeshGraphNet, MeshGraphKAN, BiStrideMeshGraphNet

num_nodes = 100
num_edges = 300
edge_index = torch.randint(0, num_nodes, (2, num_edges))
node_features = torch.randn(num_nodes, 4)
edge_features = torch.randn(num_edges, 3)

# 创建 PyG Data 对象
graph = Data(edge_index=edge_index)

# --- Base MeshGraphNet ---
model = MeshGraphNet(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    mlp_activation_fn="relu",
    aggregation="sum",
)

node_outputs = model(node_features, edge_features, graph)
print("Base MeshGraphNet Output Shape:", node_outputs.shape)

# --- HybridMeshGraphNet ---
# HybridMeshGraphNet 需要两组不同的边：mesh edges 和 world edges
num_mesh_edges = 300
num_world_edges = 300

# 创建两个不同的边索引
mesh_edge_index = torch.randint(0, num_nodes, (2, num_mesh_edges))
world_edge_index = torch.randint(0, num_nodes, (2, num_world_edges))

# 合并两个边索引（HybridMeshGraphNet 内部会处理它们）
combined_edge_index = torch.cat([mesh_edge_index, world_edge_index], dim=1)

# 创建包含合并边的图
hybrid_graph = Data(edge_index=combined_edge_index)

# 两组边的特征
mesh_edge_features = torch.randn(num_mesh_edges, 3)
world_edge_features = torch.randn(num_world_edges, 3)

model_hybrid = HybridMeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2)
node_outputs_hybrid = model_hybrid(node_features, mesh_edge_features, world_edge_features, hybrid_graph)
print("HybridMeshGraphNet Output Shape:", node_outputs_hybrid.shape)

# --- MeshGraphKAN ---
model_kan = MeshGraphKAN(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    num_harmonics=5,
)
node_outputs_kan = model_kan(node_features, edge_features, graph)
print("MeshGraphKAN Output Shape:", node_outputs_kan.shape)

# # --- BiStrideMeshGraphNet ---
# ms_edges = [torch.randint(0, 100, (2, 50)), torch.randint(0, 100, (2, 25))]
# ms_ids = [torch.arange(100), torch.arange(50)]
# model_bistride = BiStrideMeshGraphNet(
#     input_dim_nodes=4, input_dim_edges=3, output_dim=2, processor_size=10
# )
# node_outputs_bistride = model_bistride(
#     node_features, edge_features, graph, ms_edges=ms_edges, ms_ids=ms_ids
# )
# print("BiStrideMeshGraphNet Output Shape:", node_outputs_bistride.shape)

# # --- BiStrideMeshGraphNet ---
# # BiStrideMeshGraphNet 需要节点位置信息
# node_positions = torch.randn(num_nodes, 3)  # 3D 位置坐标

# # 创建包含位置信息的图
# graph_with_pos = Data(edge_index=edge_index, pos=node_positions)

# ms_edges = [torch.randint(0, 100, (2, 50)), torch.randint(0, 100, (2, 25))]
# ms_ids = [torch.arange(100), torch.arange(50)]

# model_bistride = BiStrideMeshGraphNet(
#     input_dim_nodes=4, input_dim_edges=3, output_dim=2, processor_size=10
# )
# node_outputs_bistride = model_bistride(
#     node_features, edge_features, graph_with_pos, ms_edges=ms_edges, ms_ids=ms_ids
# )
# print("BiStrideMeshGraphNet Output Shape:", node_outputs_bistride.shape)

# --- BiStrideMeshGraphNet ---
print("\n⚠ BiStrideMeshGraphNet 需要特殊的多尺度图结构")
print("文档中的简单示例无法正常工作")
print("建议：在实际应用中使用 BistrideMultiLayerGraphDataset\n")

# 如果仍想尝试（可能会报错）：
try:
    # BiStrideMeshGraphNet 需要节点位置信息
    node_positions = torch.randn(num_nodes, 3)
    graph_with_pos = Data(edge_index=edge_index, pos=node_positions)
    
    ms_edges = [torch.randint(0, 100, (2, 50)), torch.randint(0, 100, (2, 25))]
    ms_ids = [torch.arange(100), torch.arange(50)]
    
    model_bistride = BiStrideMeshGraphNet(
        input_dim_nodes=4, input_dim_edges=3, output_dim=2, processor_size=10
    )
    
    # 关键：传递 graph_with_pos，不是 edge_index
    node_outputs_bistride = model_bistride(
        node_features, edge_features, graph_with_pos, 
        ms_edges=ms_edges, ms_ids=ms_ids
    )
    print("✓ BiStrideMeshGraphNet Output Shape:", node_outputs_bistride.shape)
except RuntimeError as e:
    print(f"✗ BiStrideMeshGraphNet 报错（预期的）: {e}")
    print("  原因：需要使用 BistrideMultiLayerGraphDataset 生成正确的多尺度结构")

print("\n" + "="*60)
print("总结：NVIDIA 文档中的示例代码存在以下问题：")
print("="*60)
print("1. ✗ 传递 edge_index (Tensor) → 应该传递 graph (Data 对象)")
print("2. ✗ HybridMGN 的边结构不正确 → 需要合并 mesh 和 world 边")
print("3. ✗ BiStrideMGN 的简单示例无法运行 → 需要专门的数据集类")
print("\n✓ 实际验证可用的模型：")
print("  - MeshGraphNet")
print("  - HybridMeshGraphNet (修正后)")
print("  - MeshGraphKAN")