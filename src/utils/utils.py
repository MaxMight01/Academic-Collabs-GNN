from torch_geometric.utils import negative_sampling

def get_positive_edges(data):
    return data.edge_index.t()

def get_negative_edges(data, num_neg_samples=None):
    return negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples or data.edge_index.size(1)
    ).t()