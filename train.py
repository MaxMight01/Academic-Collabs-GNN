import torch
from src.data.build_graph import build_graph
from src.model.train import train

if __name__ == "__main__":
    data = torch.load("data/processed/graph_data.pt", weights_only=False)
    train(data)