import torch
import dgl
import physicsnemo
print(torch.__version__)
# print(dir(physicsnemo.models.meshgraphnet))
#from physicsnemo.models.meshgraphnet import MeshGraphNet, HybridMeshGraphNet, MeshGraphKAN, BiStrideMeshGraphNet
from physicsnemo.models.meshgraphnet import MeshGraphNet, MeshGraphKAN, BiStrideMeshGraphNet
# --- DGL Example (Base MeshGraphNet) ---

# Create a toy graph and random features
# In a real application, these would come from your mesh data
num_nodes = 100
num_edges = 300
edge_index = torch.randint(0, num_nodes, (2, num_edges))
node_features = torch.randn(num_nodes, 4)  # [N, input_dim_nodes]
edge_features = torch.randn(num_edges, 3)  # [E, input_dim_edges]

# 创建 DGL 图对象（Docker 容器中的 physicsnemo 版本需要 DGL 格式）
dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

# Instantiate the base model
model = MeshGraphNet(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    mlp_activation_fn="relu",
    aggregation="sum",
)

# Run a forward pass
node_outputs = model(node_features, edge_features, dgl_graph)  # [N, 2]
print("Base MeshGraphNet Output Shape:", node_outputs.shape)
# Output: Base MeshGraphNet Output Shape: torch.Size([100, 2])

# --- HybridMeshGraphNet Example ---
# The Hybrid model requires separate features for mesh and world edges
mesh_edge_features = torch.randn(edge_index.size(1), 3)
world_edge_features = torch.randn(edge_index.size(1), 3)

# model_hybrid = HybridMeshGraphNet(input_dim_nodes=4, input_dim_edges=3, output_dim=2)
# node_outputs_hybrid = model_hybrid(node_features, mesh_edge_features, world_edge_features, edge_index)
# print("HybridMeshGraphNet Output Shape:", node_outputs_hybrid.shape)
# Output: HybridMeshGraphNet Output Shape: torch.Size([100, 2])

# --- MeshGraphKAN Example ---
model_kan = MeshGraphKAN(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=10,
    num_harmonics=5,  # KAN-specific parameter
)
node_outputs_kan = model_kan(node_features, edge_features, dgl_graph)
print("MeshGraphKAN Output Shape:", node_outputs_kan.shape)
# Output: MeshGraphKAN Output Shape: torch.Size([100, 2])

# --- BiStrideMeshGraphNet Example ---
# This model requires a pre-computed graph pyramid and node positions
# BiStrideMeshGraphNet 需要使用 BistrideMultiLayerGraphDataset 来生成正确的多层图结构
try:
    from physicsnemo.datapipes.gnn.bsms import BistrideMultiLayerGraphDataset
    from torch_geometric.data import Data as PyGData
    
    # 创建 PyG 格式的图（BistrideMultiLayerGraphDataset 需要 PyG 格式）
    node_pos = torch.randn(num_nodes, 3)  # 3D 位置坐标
    pyg_graph_bistride = PyGData(
        edge_index=edge_index,
        pos=node_pos,
        x=node_features,
        edge_attr=edge_features
    )
    
    # 使用 BistrideMultiLayerGraphDataset 生成多层图结构
    num_layers = 2
    bistride_dataset = BistrideMultiLayerGraphDataset([pyg_graph_bistride], num_layers)
    sample = bistride_dataset[0]
    
    # 提取图和多尺度边、ID
    graph_bistride = sample["graph"]
    ms_edges = sample["ms_edges"]
    ms_ids = sample["ms_ids"]
    
    model_bistride = BiStrideMeshGraphNet(
        input_dim_nodes=4, 
        input_dim_edges=3, 
        output_dim=2, 
        processor_size=10,
        num_layers_bistride=num_layers
    )
    node_outputs_bistride = model_bistride(
        graph_bistride.x, graph_bistride.edge_attr, graph_bistride, 
        ms_edges=ms_edges, ms_ids=ms_ids
    )
    print("BiStrideMeshGraphNet Output Shape:", node_outputs_bistride.shape)
    # Output: BiStrideMeshGraphNet Output Shape: torch.Size([100, 2])
except ImportError as e:
    print(f"跳过 BiStrideMeshGraphNet 示例: {e}")
    print("需要安装 torch_geometric 和 torch_scatter")