import torch
import json
import argparse
from pathlib import Path
from src.model.train import train
from src.model.evaluate import evaluate
from src.analysis.visualise import EmbeddingVisualizer
from src.utils.utils import log_training_run

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the collaboration prediction model.")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing train/val/test .pt files')
    parser.add_argument('--plot_dir', type=str, default='data/plots', help='Directory to save visualisations')
    parser.add_argument('--log_dir', type=str, default='data/log', help='Directory to save logs')
    parser.add_argument('--config', type=str, default='src/config.json', help='Path to the config JSON file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, help='Hidden layer dimension')
    parser.add_argument('--layers', type=int, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, help='Dropout probability')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r') as file:
        config = json.load(file)

    lr = args.lr if args.lr is not None else config['learning_rate']
    epochs = args.epochs if args.epochs is not None else config['num_epochs']
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else config['hidden_dim']
    layers = args.layers if args.layers is not None else config['num_layers']
    dropout = args.dropout if args.dropout is not None else config['dropout']

    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / "train_data.pt", weights_only=False)
    val_data = torch.load(data_dir / "val_data.pt", weights_only=False)
    test_data = torch.load(data_dir / "test_data.pt", weights_only=False)

    model, predictor, final_embeddings = train(
        data=train_data,
        val_data=val_data,
        epochs=epochs,
        hidden_channels=hidden_dim,
        lr=lr,
        layers=layers,
        dropout=dropout
    )

    metrics = evaluate(model, predictor, test_data, True)
    hyperparams = {
        "learning_rate": lr,
        "epochs": epochs,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "dropout": dropout
    }

    log_dir = Path(args.log_dir)
    log_training_run(metrics, hyperparams, log_dir)
    print("Metadata saved.\n")

    viz = EmbeddingVisualizer(embeddings=final_embeddings, data=train_data, predictor=predictor)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    viz.visualise_pca(plot_dir / "pca.png")
    viz.visualise_tsne(plot_dir / "tsne.png")
    viz.visualise_tsne_institutions(plot_dir / "tsne_inst.png")
    print("Plots saved\n.")
    viz.analyze_feature_influence()
