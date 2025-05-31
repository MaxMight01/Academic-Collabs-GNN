import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from src.utils.utils import get_negative_edges, get_positive_edges

@torch.no_grad()
def evaluate(model, predictor, data, rtrn=False):
    model.eval()
    device = next(model.parameters()).device

    z = model(data.x.to(device), data.edge_index.to(device))

    pos_edge_index = get_positive_edges(data)
    neg_edge_index = get_negative_edges(data)

    pos_scores = predictor(z[pos_edge_index[:, 0]], z[pos_edge_index[:, 1]]).sigmoid()
    neg_scores = predictor(z[neg_edge_index[:, 0]], z[neg_edge_index[:, 1]]).sigmoid()

    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).numpy()

    auc  = roc_auc_score(labels, scores)
    pred = (scores > 0.5).astype(int)
    acc  = accuracy_score(labels, pred)
    f1   = f1_score(labels, pred)

    print(f"VAL →  AUC: {auc:.4f}   ACC: {acc:.4f}   F1: {f1:.4f}")

    if rtrn:
        return {"AUC": auc, "Accuracy": acc, "F1": f1}