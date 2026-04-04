import torch
# import dgl
from torch_geometric.data import Data

# Node indices that define a simple, 3-node, 2-edge directed graph:
src = [0, 1]
dst = [1, 2]
node_features = torch.tensor([[0.], [1.], [2.]])

# DGL:
# graph_dgl = dgl.graph((src, dst))
# graph_dgl.ndata["x"] = node_features

# PyG:
edge_index = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)
graph_pyg = Data(x=node_features, edge_index=edge_index)

# Alternative approach:
# graph_pyg = Data(edge_index=edge_index)
# graph_pyg.x = node_features

# print(graph_dgl)
print(graph_pyg)