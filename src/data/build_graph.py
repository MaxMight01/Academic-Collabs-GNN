import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

def load_data(raw_dir):
    authors = pd.read_csv(Path(raw_dir) / 'authors.csv')
    edges = pd.read_csv(Path(raw_dir) / 'coauthorship_edges.csv')
    return authors, edges

def preprocess_nodes(authors):
    mlb = MultiLabelBinarizer()
    institutions_encoded = mlb.fit_transform(eval(x) if isinstance(x, str) else [] for x in authors['institutions'])

    node_features = torch.tensor(
        pd.concat([
            pd.DataFrame(institutions_encoded),
            authors[['publications_count', 'citation_count']]
        ], axis=1).values, dtype=torch.float
    )
    id_map = {aid: idx for idx, aid in enumerate(authors['id'])}
    return node_features, id_map

def preprocess_edges(edges, id_map):
    edge_index = torch.tensor([
        [id_map[src], id_map[tgt]] for src, tgt in zip(edges['source'], edges['target'])
    ], dtype=torch.long).t().contiguous()
    return edge_index

def build_graph(raw_dir, processed_path):
    authors, edges = load_data(raw_dir)
    node_features, id_map = preprocess_nodes(authors)
    edge_index = preprocess_edges(edges, id_map)
    data = Data(x=node_features, edge_index=edge_index)

    torch.save(data, processed_path)
    print(f"Graph saved to {processed_path}")

if __name__ == "__main__":
    build_graph("data/raw", "data/processed/graph_data.pt")
