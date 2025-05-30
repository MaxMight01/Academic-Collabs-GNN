import torch
from torch_geometric.utils import negative_sampling
from src.model.sage_link_predictor import GraphSAGE, LinkPredictor
from src.model.evaluate import evaluate
from src.utils.utils import get_positive_edges, get_negative_edges

def train(data, val_data=None, epochs=50, hidden_channels=64, lr=0.01, layers=2, dropout=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(in_channels=data.num_node_features, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=layers, dropout=dropout).to(device)
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

        criterion = torch.nn.BCEWithLogitsLoss()

        pos_labels = torch.ones(pos_edge_index.size(0), device=device)
        neg_labels = torch.zeros(neg_edge_index.size(0), device=device)

        pos_pred = predictor(z[pos_edge_index[:, 0]], z[pos_edge_index[:, 1]])
        neg_pred = predictor(z[neg_edge_index[:, 0]], z[neg_edge_index[:, 1]])

        loss = criterion(pos_pred, pos_labels) + criterion(neg_pred, neg_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

        if val_data is not None:
            val_data = val_data.to(device)
            evaluate(model, predictor, val_data)

    return model, predictor