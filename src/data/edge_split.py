import torch
from pathlib import Path
from torch_geometric.transforms import RandomLinkSplit

def split_graph_edges(processed_dir):
    graph_path = processed_dir / "graph_data.pt"
    data = torch.load(graph_path, weights_only=False)

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=False
    )

    train_data, val_data, test_data = transform(data)

    torch.save(train_data, processed_dir / "train_data.pt")
    torch.save(val_data, processed_dir / "val_data.pt")
    torch.save(test_data, processed_dir / "test_data.pt")
    print("Graph successfully split and saved.")

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_graph_edges(processed_dir)
