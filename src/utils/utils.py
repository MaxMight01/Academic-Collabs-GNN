import json
from pathlib import Path
from datetime import datetime
from torch_geometric.utils import negative_sampling

def get_positive_edges(data):
    return data.edge_index.t()

def get_negative_edges(data, num_neg_samples=None):
    return negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples or data.edge_index.size(1)
    ).t()

def log_training_run(metrics, hyperparams, log_dir="data/logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.json"
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_data = {
        "timestamp": timestamp,
        "hyperparameters": hyperparams,
        "metrics": metrics
    }

    with open(log_path / filename, "w") as f:
        json.dump(log_data, f, indent=4)