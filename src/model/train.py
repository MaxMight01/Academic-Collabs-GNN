# src/model/train.py
import torch
import random
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from src.model.sage_link_predictor import GraphSAGE, LinkPredictor

def get_positive_edges(data):
    return data.edge_index.t()

def get_negative_edges(data, num_neg_samples=None):
    return negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg_samples or data.edge_index.size(1)
    ).t()

def train(data, epochs=50, hidden_channels=64, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(data.num_node_features, hidden_channels, hidden_channels).to(device)
    predictor = LinkPredictor(hidden_channels).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)

        # Positive edges
        pos_edge_index = get_positive_edges(data)
        neg_edge_index = get_negative_edges(data)

        criterion = nn.BCEWithLogitsLoss()

        pos_labels = torch.ones(pos_edge_index.size(0), device=device)
        neg_labels = torch.zeros(neg_edge_index.size(0), device=device)

        pos_pred = predictor(z[pos_edge_index[:, 0]], z[pos_edge_index[:, 1]])
        neg_pred = predictor(z[neg_edge_index[:, 0]], z[neg_edge_index[:, 1]])

        loss = criterion(pos_pred, pos_labels) + criterion(neg_pred, neg_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
