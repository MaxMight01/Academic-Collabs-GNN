import torch
from torch_geometric.transforms import RandomLinkSplit

data = torch.load("data/processed/graph_data.pt", weights_only=False)

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    split_labels=False
)
train_data, val_data, test_data = transform(data)

torch.save(train_data, "data/processed/train_data.pt")
torch.save(val_data, "data/processed/val_data.pt")
torch.save(test_data, "data/processed/test_data.pt")