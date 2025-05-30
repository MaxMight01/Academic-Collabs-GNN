import torch
import json
from src.model.train import train
from src.model.evaluate import evaluate
from src.analysis.visualise import EmbeddingVisualizer

if __name__ == "__main__":
    with open('src/config.json', 'r') as file:
        config = json.load(file)

    lr = config['learning_rate']
    epochs = config['num_epochs']
    hidden_dim = config['hidden_dim']
    layers = config['num_layers']
    dropout = config['dropout']

    train_data = torch.load("data/processed/train_data.pt", weights_only=False)
    val_data = torch.load("data/processed/val_data.pt", weights_only=False)
    model, predictor, final_embeddings = train(data=train_data, val_data=val_data, epochs=epochs, hidden_channels=hidden_dim, lr=lr, layers=layers, dropout=dropout)

    test_data = torch.load("data/processed/test_data.pt", weights_only=False)
    evaluate(model, predictor, test_data)

    # visualise_tsne_with_institutions(final_embeddings=final_embeddings, data=train_data)

    viz = EmbeddingVisualizer(embeddings=final_embeddings, data=train_data, predictor=predictor)
    viz.visualise_pca("data/plots/pca.png")
    viz.visualise_tsne("data/plots/tsne.png")
    viz.visualise_tsne_institutions("data/plots/tsne_inst.png")
    viz.analyze_feature_influence()
